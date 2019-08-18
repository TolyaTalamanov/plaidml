// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/emitcm.h"

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "base/util/error.h"
#include "tile/lang/exprtype.h"
#include "tile/lang/fpconv.h"

#include "base/util/env.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

void Emit::Visit(const sem::SubscriptLVal& n) {
  if (in_write_statement) {
    if (!use_global_id) {
      int stride = GetLocalIndexStride(n.offset);
      if (stride > 1) {
        n.ptr->Accept(*this);
        emit(", ");
        n.offset->Accept(*this);
        emit(", ");
        emit("element_offset_");
        emit(std::to_string(stride));
        return;
      }
    }
    n.ptr->Accept(*this);
    emit(", sizeof(");
    emitType(write_type);
    emit(") * ");
    n.offset->Accept(*this);
    return;
  }

  auto s = GetGlobalVarWithOffset(n);
  if (s.length() > 0 && input_replace_map.find(s) != input_replace_map.end()) {
    emit(input_replace_map[s]);
    return;
  }

  if (is_sub_group_broadcast_first_val) {
    is_sub_group_broadcast_first_val = false;
    n.ptr->Accept(*this);
    emit("(");
    n.offset->Accept(*this);
    auto is_lookup_lval = std::dynamic_pointer_cast<sem::LookupLVal>(n.ptr);
    if (large_sparse_vactor.find(is_lookup_lval->name) == large_sparse_vactor.end()) {
      emit(" * ");
      emit(vector_size);
    }
    return;
  }
  n.ptr->Accept(*this);
  auto is_lookup_lval = std::dynamic_pointer_cast<sem::LookupLVal>(n.ptr);

  if (in_read_statement || vector_stride_map.find(GetLValueName(n.ptr)) == vector_stride_map.end() ||
      vector_stride_map[GetLValueName(n.ptr)] >= 1) {
    emit(".select<");
    emit(vector_size);
    emit(", 1>");
  }
  emit("(");
  n.offset->Accept(*this);
  if (large_sparse_vactor.find(is_lookup_lval->name) == large_sparse_vactor.end()) {
    emit(" * ");
    emit(vector_size);
  }
  emit(")");
}

void Emit::Visit(const sem::LoadExpr& n) {
  auto ty = TypeOf(n.inner);
  auto s = GetGlobalVarWithOffset(n.inner);
  if (s.length() > 0 && input_replace_map.find(s) != input_replace_map.end()) {
    emit(input_replace_map[s]);
    return;
  }
  auto inner = std::dynamic_pointer_cast<sem::SubscriptLVal>(n.inner);
  if (inner && GetGlobalVarWithOffset(inner).size() > 0) {
    if (!use_global_id) {
      int stride = GetLocalIndexStride(inner->offset);
      if (stride > 1) {
        inner->ptr->Accept(*this);
        emit(", ");
        inner->offset->Accept(*this);
        emit(", ");
        emit("element_offset_");
        emit(std::to_string(stride));
        return;
      }
    }
    inner->ptr->Accept(*this);
    emit(", sizeof(");
    emitType(ty);
    emit(") * ");
    inner->offset->Accept(*this);
  } else {
    n.inner->Accept(*this);
  }
}

void Emit::assign_global_var_to_temp(const sem::ExprPtr& e) {
  auto result_map = GetGlobalLoadExprMap(e);
  for (auto result : result_map) {
    std::string temp_val = "cm_temp" + std::to_string(temp_var_num);
    temp_var_num++;
    auto type = TypeOf(result.first->inner);
    EmitVector(type, vector_size, temp_val);
    emit(";\n");

    auto s = GetGlobalVarWithOffset(result.first->inner);
    if (s.length() > 0 && input_replace_map.find(s) != input_replace_map.end()) {
      emitTab();
      emit(temp_val);
      emit(" = ");
      emit(input_replace_map[s]);
      emit(";\n");
    } else {
      emitTab();
      emit("_read(");
      in_read_statement = true;
      result.first->Accept(*this);
      emit(", ");
      emit(temp_val);
      emit(");\n");
      in_read_statement = false;
      input_replace_map[result.second] = temp_val;
    }
  }
}

inline std::string c_dtype(const DataType& dt) {
  std::string base;
  switch (dt) {
    case DataType::BOOLEAN:
      base = "(bool)";
      break;
    case DataType::INT8:
      base = "(char)";
      break;
    case DataType::INT16:
      base = "(short)";
      break;
    case DataType::INT32:
      base = "(int)";
      break;
    case DataType::INT64:
      base = "(long)";
      break;
    case DataType::UINT8:
      base = "(uchar)";
      break;
    case DataType::UINT16:
      base = "(ushort)";
      break;
    case DataType::UINT32:
      base = "(uint)";
      break;
    case DataType::UINT64:
      base = "(ulong)";
      break;
    case DataType::FLOAT16:
      base = "(half)";
      break;
    case DataType::FLOAT32:
      base = "(float)";
      break;
    case DataType::FLOAT64:
      base = "(double)";
      break;
    default:
      throw std::runtime_error("Invalid tile type");
  }
  return base;
}

void Emit::SingleElementWrite(const sem::LValPtr& lhs, const sem::ExprPtr& rhs) {
  emitTab();
  auto ty_lhs = TypeOf(lhs);
  switch (ty_lhs.dtype) {
    case DataType::INT8:
    case DataType::UINT8:
    case DataType::INT16:
    case DataType::UINT16:
    case DataType::FLOAT16:
      emit("_write_single_element(");
      break;
    case DataType::INT32:
    case DataType::UINT32:
    case DataType::FLOAT32:
      emit("_write_atomic_single_dword(");
      break;
    case DataType::INT64:
      emit("_write_atomic_single_long(");
      break;
    default:
      throw std::runtime_error("SingleElementWrite: this data type is not supported!");
  }
  in_write_statement = true;
  write_type = ty_lhs;
  lhs->Accept(*this);
  in_write_statement = false;
  emit(", ");
  auto ty_rhs = TypeOf(rhs);
  if (c_dtype(ty_lhs.dtype) != c_dtype(ty_rhs.dtype)) {
    emit(c_dtype(ty_lhs.dtype));
  }
  rhs->Accept(*this);
  if (IsVector(rhs)) {
    emit("(0)");
  }
  emit(");\n");
}

void Emit::Visit(const sem::StoreStmt& n) {
  auto ty_lhs = TypeOf(n.lhs);

  auto is_lhs_global = GetGlobalVarWithOffset(n.lhs).size() > 0;
  auto rhs_load_exp = std::dynamic_pointer_cast<sem::LoadExpr>(n.rhs);
  bool is_rhs_global = (rhs_load_exp != nullptr) && GetGlobalVarWithOffset(rhs_load_exp->inner).size() > 0;

  auto rhs_int_const = std::dynamic_pointer_cast<sem::IntConst>(n.rhs);
  if (is_lhs_global && rhs_int_const) {
    emitTab();
    emit("_write_atomic_single_dword(");
    in_write_statement = true;
    write_type = ty_lhs;
    n.lhs->Accept(*this);
    in_write_statement = false;
    emit(", ");
    n.rhs->Accept(*this);
    emit(");\n");
    return;
  }

  if (is_lhs_global && is_rhs_global) {
    assign_global_var_to_temp(n.rhs);

    if (use_global_id) {
      SingleElementWrite(n.lhs, n.rhs);
    } else {
      emitTab();
      emit("_write(");
      in_write_statement = true;
      write_type = ty_lhs;
      n.lhs->Accept(*this);
      in_write_statement = false;
      emit(", ");
      n.rhs->Accept(*this);
      emit(");\n");
    }
  }

  if (is_lhs_global && !is_rhs_global) {
    auto p = std::dynamic_pointer_cast<sem::SubscriptLVal>(n.lhs);
    if (p == nullptr) throw std::runtime_error("StoreStmt lhs is not SubscriptLVal!");

    auto cond_expr = std::dynamic_pointer_cast<sem::CondExpr>(n.rhs);
    if (cond_expr) {
      assign_global_var_to_temp(n.rhs);

      std::string temp_var = "cm_temp" + std::to_string(temp_var_num);
      temp_var_num++;
      EmitVector(ty_lhs, vector_size, temp_var);
      emit(";\n");

      emitTab();
      emit(temp_var);
      emit(".");
      n.rhs->Accept(*this);
      emit(";\n");

      emitTab();
      emit("_write(");
      in_write_statement = true;
      write_type = ty_lhs;
      n.lhs->Accept(*this);
      in_write_statement = false;
      emit(", ");
      emit(temp_var);
      emit(");\n");

      return;
    }

    auto binary_expr = std::dynamic_pointer_cast<sem::BinaryExpr>(n.rhs);
    if (binary_expr) {
      assign_global_var_to_temp(n.rhs);

      if (use_global_id) {
        SingleElementWrite(n.lhs, n.rhs);
      } else {
        emitTab();
        emit("_write(");
        in_write_statement = true;
        write_type = ty_lhs;
        n.lhs->Accept(*this);
        in_write_statement = false;
        emit(", ");
        n.rhs->Accept(*this);
        emit(");\n");
      }
      return;
    }

    if (use_global_id) {
      SingleElementWrite(n.lhs, n.rhs);
    } else {
      assign_global_var_to_temp(n.rhs);
      emitTab();
      emit("_write(");
      in_write_statement = true;
      write_type = ty_lhs;
      n.lhs->Accept(*this);
      in_write_statement = false;
      emit(", ");
      n.rhs->Accept(*this);
      emit(");\n");
    }
  }

  if (!is_lhs_global && is_rhs_global) {
    auto rhs_load_exp = std::dynamic_pointer_cast<sem::LoadExpr>(n.rhs);
    if (rhs_load_exp) {
      emitTab();
      emit("_read(");
      in_read_statement = true;
      rhs_load_exp->Accept(*this);
      emit(", ");
      n.lhs->Accept(*this);
      emit(");\n");
      in_read_statement = false;

      auto stride = GetLocalIndexStride(rhs_load_exp);
      vector_stride_map[GetLValueName(n.lhs)] = stride;
    }
  }

  if (!is_lhs_global && !is_rhs_global) {
    emitTab();
    n.lhs->Accept(*this);
    auto cond_exp = std::dynamic_pointer_cast<sem::CondExpr>(n.rhs);
    auto select_exp = std::dynamic_pointer_cast<sem::SelectExpr>(n.rhs);
    if (cond_exp || select_exp) {
      emit(".");
      n.rhs->Accept(*this);
      emit(";\n");
      return;
    }

    emit(" = ");
    n.rhs->Accept(*this);
    emit(";\n");
  }
}

void Emit::Visit(const sem::DeclareStmt& n) {
  in_declare_stmt = true;
  sem::Type ty = n.type;
  sem::Type init_type;
  if (n.init) {
    init_type = TypeOf(n.init);
  }

  if (ty.base == sem::Type::VALUE) {
    if (ty.dtype == DataType::FLOAT16) {
      // ty.dtype = DataType::FLOAT32;
    } else if (ty.dtype == DataType::BOOLEAN) {
      if (n.init) {
        ty.dtype = lang::Promote({init_type}).dtype;
        if (ty.dtype == DataType::BOOLEAN) {
          // If the initializer was booleans, make it INT8.
          ty.dtype = DataType::INT8;
        }
      } else {
        // Assume that this is being initialized from an inter-kernel
        // boolean tensor -- which, in cm, we represent as INT8.
        ty.dtype = DataType::INT8;
      }
    }
  }

  if (n.init) {
    if (ty.base == sem::Type::INDEX) {
      emit("// map=");

      for (auto r : tv.index_stride_map) {
        emit(r.first);
        emit(": ");
        emit(std::to_string(r.second));
        emit("    ");
      }
      emit("\n");

      int stride = GetLocalIndexStride(n.init);
      tv.index_stride_map[n.name] = stride;

      if (stride > 1) {
        std::string vname = "element_offset_" + std::to_string(stride);
        if (element_offset_vector.find(vname) == element_offset_vector.end()) {
          if (!use_global_id) {
            emitTab();
            emit("cm_vector(");
            emit(vname);
            emit(", uint, ");
            emit(vector_size);
            emit(", 0, ");
            emit(std::to_string(stride));
            emit(");\n");
            element_offset_vector.insert(vname);
          }
        }
      }
      emit("// use_global_id=");
      if (use_global_id) {
        emit("True ");
      } else {
        emit("False ");
      }
      emit("\n");
    }

    auto load_exp = std::dynamic_pointer_cast<sem::LoadExpr>(n.init);
    if (load_exp) {
      if (!IsVector(load_exp->inner) && GetGlobalVarWithOffset(load_exp->inner).size() == 0) {
        emitTab();
        emitType(ty);
        emit(" ");
        emit(n.name);

        if (ty.base == sem::Type::INDEX) {
          if (depend_on_local_id(n.init)) {
            dependent_index.insert(n.name);
          }
        }
        if (n.init) {
          emit(" = ");
          n.init->Accept(*this);
        }
        emit(";\n");

        CheckValidType(ty);
        scope_->Bind(n.name, ty);
        in_declare_stmt = false;
        return;
      }
      EmitVector(ty, vector_size, n.name);
      emit(";\n");

      if (GetGlobalVarWithOffset(load_exp->inner).size() > 0) {
        emitTab();
        emit("_read(");
        in_read_statement = true;
        load_exp->Accept(*this);
        emit(", ");
        emit(n.name);
        emit(");\n");
        in_read_statement = false;
      } else {
        emitTab();
        emit(n.name);
        emit(" = ");
        load_exp->Accept(*this);
        emit(";\n");
      }
      CheckValidType(ty);
      scope_->Bind(n.name, ty);
      in_declare_stmt = false;
      return;
    }

    auto cast_exp = std::dynamic_pointer_cast<sem::CastExpr>(n.init);
    if (cast_exp) {
      auto load_exp = std::dynamic_pointer_cast<sem::LoadExpr>(cast_exp->val);
      if (load_exp) {
        EmitVector(ty, vector_size, n.name);
        emit(" = ");
        load_exp->Accept(*this);
        emit(";\n");
        CheckValidType(ty);
        scope_->Bind(n.name, ty);
        in_declare_stmt = false;
        return;
      }
    }

    auto cond_exp = std::dynamic_pointer_cast<sem::CondExpr>(n.init);
    if (cond_exp) {
      if (IsVector(cond_exp->tcase) || IsVector(cond_exp->fcase) || IsVector(cond_exp->cond)) {
        EmitVector(ty, vector_size, n.name);
        emit(";\n");

        emitTab();
        emit(n.name);
        emit(".");
        n.init->Accept(*this);
        emit(";\n");
        CheckValidType(ty);
        scope_->Bind(n.name, ty);
        in_declare_stmt = false;
        return;
      } else {
        emitTab();
        emitType(ty);
        emit(" ");
        emit(n.name);
        emit(".");
        n.init->Accept(*this);
        emit(";\n");
        CheckValidType(ty);
        scope_->Bind(n.name, ty);
        in_declare_stmt = false;
        return;
      }
    }

    auto select_exp = std::dynamic_pointer_cast<sem::SelectExpr>(n.init);
    if (select_exp) {
      if (IsVector(cond_exp->tcase) || IsVector(cond_exp->fcase) || IsVector(cond_exp->cond)) {
        EmitVector(ty, vector_size, n.name);
        emit(";\n");

        emitTab();
        emit(n.name);
        emit(".");
        n.init->Accept(*this);
        emit(";\n");
        CheckValidType(ty);
        scope_->Bind(n.name, ty);
        in_declare_stmt = false;
        return;
      } else {
        emitTab();
        emitType(ty);
        emit(" ");
        emit(n.name);
        emit(".");
        n.init->Accept(*this);
        emit(";\n");
        CheckValidType(ty);
        scope_->Bind(n.name, ty);
        in_declare_stmt = false;
        return;
      }
    }

    auto binary_exp = std::dynamic_pointer_cast<sem::BinaryExpr>(n.init);
    if (binary_exp) {
      if (IsVector(binary_exp->lhs) || IsVector(binary_exp->rhs)) {
        if (binary_exp->op == ">" || binary_exp->op == "<" || binary_exp->op == ">=" || binary_exp->op == "<=" ||
            binary_exp->op == "==" || binary_exp->op == "!=") {
          EmitVector("char", vector_size, n.name);
          ty.dtype = DataType::INT8;
          scope_->Bind(n.name, ty);
        } else {
          EmitVector(ty, vector_size, n.name);
          scope_->Bind(n.name, ty);
        }

        emit(" = ");
        n.init->Accept(*this);
        emit(";\n");
        CheckValidType(ty);
        // scope_->Bind(n.name, ty);
        in_declare_stmt = false;
        return;
      }
    }

    auto call_expr = std::dynamic_pointer_cast<sem::CallExpr>(n.init);
    if (call_expr) {
      for (auto val : call_expr->vals) {
        if (IsVector(val)) {
          EmitVector(ty, vector_size, n.name);
          emit(" = ");

          call_expr->Accept(*this);
          emit(";\n");
          CheckValidType(ty);
          scope_->Bind(n.name, ty);
          in_declare_stmt = false;
          return;
        }
      }
    }

    auto unary_expr = std::dynamic_pointer_cast<sem::UnaryExpr>(n.init);
    if (unary_expr && IsVector(unary_expr)) {
      EmitVector(ty, vector_size, n.name);
      emit(" = ");

      unary_expr->Accept(*this);
      emit(";\n");
      CheckValidType(ty);
      scope_->Bind(n.name, ty);
      in_declare_stmt = false;
      return;
    }
  }

  if (n.type.array) {
    if (n.type.array >= 128) {
      large_sparse_vactor.insert(n.name);
      EmitVector(ty, std::to_string(n.type.array), n.name);
      // throw std::runtime_error("cm vector exceeds maximum supported size");
    } else {
      EmitVector(ty, std::to_string(n.type.array) + " * " + vector_size, n.name);
    }
    emit(" = ");
    if (n.init) {
      n.init->Accept(*this);
    } else {
      emit("0");
    }
    emit(";\n");

  } else {
    emitTab();
    emitType(ty);
    emit(" ");
    emit(n.name);

    if (ty.base == sem::Type::INDEX) {
      if (depend_on_local_id(n.init)) {
        dependent_index.insert(n.name);
      }
    }
    if (n.init) {
      emit(" = ");
      n.init->Accept(*this);
    }
    emit(";\n");
  }

  CheckValidType(ty);
  scope_->Bind(n.name, ty);
  in_declare_stmt = false;
}

void Emit::Visit(const sem::CondExpr& n) {
  auto type = TypeOf(n.cond);

  emit("merge(");
  n.tcase->Accept(*this);
  emit(", ");
  n.fcase->Accept(*this);
  emit(", ");
  if (type.dtype != DataType::INT8) {
    auto load_exp = std::dynamic_pointer_cast<sem::LoadExpr>(n.cond);
    if (load_exp && IsVector(load_exp->inner)) {
      emit("vector<char,");
      emit(vector_size);
      emit(">(");
      n.cond->Accept(*this);
      emit(")");
    } else {
      n.cond->Accept(*this);
    }
  } else {
    n.cond->Accept(*this);
  }
  emit(")");
}

void Emit::Visit(const sem::SelectExpr& n) {
  auto type = TypeOf(n.cond);

  emit("merge(");
  n.tcase->Accept(*this);
  emit(", ");
  n.fcase->Accept(*this);
  emit(", ");
  if (type.dtype != DataType::INT8) {
    auto load_exp = std::dynamic_pointer_cast<sem::LoadExpr>(n.cond);
    if (load_exp && IsVector(load_exp->inner)) {
      emit("vector<char,");
      emit(vector_size);
      emit(">(");
      n.cond->Accept(*this);
      emit(")");
    } else {
      n.cond->Accept(*this);
    }
  } else {
    n.cond->Accept(*this);
  }
  emit(")");
}

void Emit::Visit(const sem::ClampExpr& n) {
  auto ty_val = TypeOf(n.val);
  auto ty_min = TypeOf(n.min);
  auto ty_max = TypeOf(n.max);

  // Align value dtypes and vector widths.
  sem::Type ty_clamp{sem::Type::VALUE};
  if (ty_val.base == sem::Type::VALUE) {
    ty_clamp.dtype = ty_val.dtype;
  } else {
    ty_clamp.dtype = DataType::INT32;
  }
  if (ty_min.vec_width != 1) {
    ty_clamp.vec_width = ty_min.vec_width;
  } else {
    ty_clamp.vec_width = ty_max.vec_width;
  }

  emit("_cmamp(");
  in_cmamp_function = true;
  auto load_expr = std::dynamic_pointer_cast<sem::LoadExpr>(n.val);
  if (load_expr) {
    auto s = GetGlobalVarWithOffset(load_expr->inner);
    if (s.length() > 0 && input_replace_map.find(s) != input_replace_map.end()) {
      emit(input_replace_map[s]);

      auto subscript_lval = std::dynamic_pointer_cast<sem::SubscriptLVal>(load_expr->inner);
      if (subscript_lval) {
        emit("(_mod(");
        subscript_lval->offset->Accept(*this);
        emit(" , 4))");
      }

      emit(", ");
      n.min->Accept(*this);
      emit(", ");
      n.max->Accept(*this);
      emit(")");
      return;
    }
  }
  n.val->Accept(*this);
  emit(", ");
  n.min->Accept(*this);
  emit(", ");
  n.max->Accept(*this);
  emit(")");
  in_cmamp_function = false;
}

void Emit::Visit(const sem::CastExpr& n) { n.val->Accept(*this); }

void Emit::Visit(const sem::CallExpr& n) {
  if (n.name == "sub_group_broadcast") {
    is_sub_group_broadcast_first_val = true;
    n.vals[0]->Accept(*this);
    is_sub_group_broadcast_first_val = false;
    emit(" + ");
    n.vals[1]->Accept(*this);
    emit(")");
    return;
  }
  auto it = FuncNameMap.find(n.name);
  if (it != FuncNameMap.end()) {
    emit(it->second);
  } else {
    // Assume this is an cm function.
    // TODO: Enumerate the set of callable functions.
    emit(n.name);
  }
  emit("(");
  for (size_t i = 0; i < n.vals.size(); i++) {
    n.vals[i]->Accept(*this);
    if (i != n.vals.size() - 1) {
      emit(", ");
    }
  }
  emit(")");
}

void Emit::Visit(const sem::IndexExpr& n) {
  switch (n.type) {
    case sem::IndexExpr::GLOBAL:
      if (one_thread_mode) {
        emit("_i" + std::to_string(n.dim));
        return;
      }
      emit("(cm_local_size(" + std::to_string(n.dim) + ")");
      emit(" * cm_group_id(" + std::to_string(n.dim) + ")");
      emit(" + cm_local_id(" + std::to_string(n.dim) + "))");
      break;
    case sem::IndexExpr::GROUP:
      emit("cm_group_id(" + std::to_string(n.dim) + ")");
      break;
    case sem::IndexExpr::LOCAL:
      if (one_thread_mode) {
        emit("_i" + std::to_string(n.dim));
        return;
      }
      if (use_global_id) {
        emit("cm_local_id(" + std::to_string(n.dim) + ")");
      } else {
        emit(vector_size);
        emit(" * cm_local_id(" + std::to_string(n.dim) + ")");
      }
      break;
    default:
      throw std::runtime_error("Invalid IndexExpr type");
  }
}

void Emit::Visit(const sem::Block& n) {
  auto previous_scope = scope_;
  lang::Scope<sem::Type> scope{scope_};
  scope_ = &scope;
  EmitC::Visit(n);
  scope_ = previous_scope;
}

void Emit::Visit(const sem::ForStmt& n) {
  auto previous_scope = scope_;
  lang::Scope<sem::Type> scope{scope_};
  scope_ = &scope;
  scope.Bind(n.var, sem::Type{sem::Type::INDEX});
  EmitC::Visit(n);
  scope_ = previous_scope;
}

void Emit::Visit(const sem::BarrierStmt& n) {}

void Emit::Visit(const sem::Function& n) {
  emit("extern \"C\" _GENX_MAIN_ ");

  if (n.subgroup_size) {
    use_global_id = false;
    vector_size = std::to_string(n.subgroup_size);
  } else {
    use_global_id = true;
    vector_size = "4";
  }

  lang::Scope<sem::Type> scope;
  scope_ = &scope;

  one_thread_mode = false;
  for (const auto& p : n.params) {
    auto ty = p.first;
    if (ty.dtype == DataType::BOOLEAN) {
      // Global booleans are stored as INT8.
      ty.dtype = DataType::INT8;
    }
    if (ty.dtype == DataType::INT8 || ty.dtype == DataType::UINT8 || ty.dtype == DataType::INT16 ||
        ty.dtype == DataType::UINT16) {
      one_thread_mode = true;
    }
    CheckValidType(ty);
    scope.Bind(p.second, ty);
    tv.global_params.insert(p.second);
    tv.vector_params.insert(p.second);
  }

  emitType(n.ret);
  emit(" ");
  emit(n.name);
  emit("(");
  bool first_param = true;
  for (const auto& p : n.params) {
    if (first_param) {
      first_param = false;
    } else {
      emit(", ");
    }
    emit("SurfaceIndex");
    emit(" ");
    emit(p.second);
  }
  emit(")\n");

  if (one_thread_mode) {
    int lsize0 = ki_.gwork[0];
    int lsize1 = ki_.gwork[1];
    int lsize2 = ki_.gwork[2];
    emit("{\n");
    ++indent_;
    emitTab();
    emit("if(cm_local_id(0) == 0 && cm_group_id(0) == 0){\n");
    ++indent_;
    emitTab();
    emit("for(int _i0=0;_i0<" + std::to_string(lsize0) + ";_i0++){\n");
    ++indent_;
    emitTab();
    emit("for(int _i1=0;_i1<" + std::to_string(lsize1) + ";_i1++){\n");
    ++indent_;
    emitTab();
    emit("for(int _i2=0;_i2<" + std::to_string(lsize2) + ";_i2++){\n");
    n.body->Accept(*this);
    emitTab();
    emit("}\n");
    --indent_;
    emitTab();
    emit("}\n");
    --indent_;
    emitTab();
    emit("}\n");
    --indent_;
    emitTab();
    emit("}\n");
    --indent_;
    emitTab();
    emit("}\n");
  } else {
    n.body->Accept(*this);
  }
  scope_ = nullptr;
}

bool Emit::depend_on_local_id(const sem::ExprPtr& init) {
  auto index_expr = std::dynamic_pointer_cast<sem::IndexExpr>(init);
  if (index_expr) {
    return (index_expr->type == sem::IndexExpr::GLOBAL || index_expr->type == sem::IndexExpr::LOCAL);
  }
  auto binary_expr = std::dynamic_pointer_cast<sem::BinaryExpr>(init);
  if (binary_expr) {
    if (!binary_expr->op.compare("/")) {
      auto index_expr = std::dynamic_pointer_cast<sem::IndexExpr>(binary_expr->lhs);
      auto int_const = std::dynamic_pointer_cast<sem::IntConst>(binary_expr->rhs);
      if (index_expr && int_const) {
        if ((index_expr->type == sem::IndexExpr::GLOBAL || index_expr->type == sem::IndexExpr::LOCAL) &&
            (int_const->value % 16) == 0)
          return false;
      }
    }
    return (depend_on_local_id(binary_expr->lhs) || depend_on_local_id(binary_expr->rhs));
  }
  auto load_expr = std::dynamic_pointer_cast<sem::LoadExpr>(init);
  if (load_expr) {
    auto lookup_lval = std::dynamic_pointer_cast<sem::LookupLVal>(load_expr->inner);
    if (lookup_lval && (dependent_index.find(lookup_lval->name) != dependent_index.end())) return true;
  }
  return false;
}

std::string Emit::GetLValueName(const sem::LValPtr& p) {
  auto pval = std::dynamic_pointer_cast<sem::LookupLVal>(p);
  if (pval) {
    return pval->name;
  }

  auto psval = std::dynamic_pointer_cast<sem::SubscriptLVal>(p);
  if (psval) {
    return GetLValueName(psval->ptr);
  }

  throw error::Unimplemented{"GetLValueName: Not Supported LValue"};
}

void Emit::CheckValidType(const sem::Type& ty) {
  if (ty.base == sem::Type::TVOID || ty.base == sem::Type::INDEX) {
    return;
  }
  if (ty.dtype == DataType::FLOAT64) {
    throw error::Unimplemented{"The device does not support 64-bit floating-point types"};
  }
}

sem::Type Emit::TypeOf(const sem::ExprPtr& p) { return lang::ExprType::TypeOf(scope_, false, true, p); }

sem::Type Emit::TypeOf(const sem::LValPtr& p) { return lang::ExprType::TypeOf(scope_, false, true, p); }

bool Emit::IsVector(const sem::ExprPtr& p) {
  tv.InitCheckVector();
  p->Accept(tv);
  return tv.CheckVector();
}

bool Emit::IsVector(const sem::LValPtr& p) {
  tv.InitCheckVector();
  p->Accept(tv);
  return tv.CheckVector();
}

bool Emit::IsVector(const sem::LValue& v) {
  tv.InitCheckVector();
  v.Accept(tv);
  return tv.CheckVector();
}

int Emit::GetLocalIndexStride(const sem::ExprPtr& p) {
  tv.InitIndexStride();
  p->Accept(tv);
  return tv.GetIndexStride();
}

int Emit::GetLocalIndexStride(const sem::LValPtr& p) {
  tv.InitIndexStride();
  p->Accept(tv);
  return tv.GetIndexStride();
}

std::string Emit::GetGlobalVarWithOffset(const sem::LValPtr& p) {
  tv.InitGlobalVarWithOffset();
  p->Accept(tv);
  return tv.GetGlobalVarWithOffset();
}

std::string Emit::GetGlobalVarWithOffset(const sem::LValue& v) {
  tv.InitGlobalVarWithOffset();
  v.Accept(tv);
  return tv.GetGlobalVarWithOffset();
}

std::map<std::shared_ptr<sem::LoadExpr>, std::string> Emit::GetGlobalLoadExprMap(const sem::ExprPtr& p) {
  tv.InitGlobalLoadExprMap();
  p->Accept(tv);
  return tv.GetGlobalLoadExprMap();
}

void Emit::EmitVector(const sem::Type& type, const std::string& size, const std::string& name) {
  emitTab();
  emit("vector<");
  emitType(type);
  emit(",");
  emit(size);
  emit("> ");
  emit(name);
  tv.vector_params.insert(name);
}

void Emit::EmitVector(const std::string& type, const std::string& size, const std::string& name) {
  emitTab();
  emit("vector<");
  emit(type);
  emit(",");
  emit(size);
  emit("> ");
  emit(name);
  tv.vector_params.insert(name);
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
