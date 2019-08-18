// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#include "tile/lang/emitc.h"
#include "tile/lang/scope.h"
#include "tile/lang/semprinter.h"

#include "tile/hal/cm/compiler.h"
#include "tile/hal/cm/travel.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

class Emit : public lang::EmitC {
 public:
  explicit Emit(lang::KernelInfo ki) : scope_{nullptr}, ki_{ki} {}

  void Visit(const sem::SubscriptLVal&) final;
  void Visit(const sem::LoadExpr&) final;
  void Visit(const sem::StoreStmt&) final;
  void Visit(const sem::DeclareStmt&) final;
  void Visit(const sem::CondExpr& n) final;
  void Visit(const sem::SelectExpr& n) final;
  void Visit(const sem::ClampExpr& n) final;
  void Visit(const sem::CastExpr&) final;
  void Visit(const sem::CallExpr&) final;
  void Visit(const sem::IndexExpr&) final;
  void Visit(const sem::Block&) final;
  void Visit(const sem::ForStmt&) final;
  void Visit(const sem::BarrierStmt&) final;
  void Visit(const sem::Function&) final;

  static size_t vsize;
  bool use_global_id = true;
  std::string vector_size = "16";

  bool comments_contains_ident = false;
  bool one_thread_mode = false;

  TravelVisitor tv;

 private:
  void CheckValidType(const sem::Type& ty);
  sem::Type TypeOf(const sem::ExprPtr& expr);
  sem::Type TypeOf(const sem::LValPtr& lvalue);
  bool IsVector(const sem::ExprPtr& p);
  bool IsVector(const sem::LValPtr& p);
  bool IsVector(const sem::LValue& v);
  int GetLocalIndexStride(const sem::LValPtr& p);
  int GetLocalIndexStride(const sem::ExprPtr& p);
  std::string GetGlobalVarWithOffset(const sem::LValPtr& p);
  std::string GetGlobalVarWithOffset(const sem::LValue& v);
  void EmitVector(const sem::Type& type, const std::string& size, const std::string& name);
  void EmitVector(const std::string& type, const std::string& size, const std::string& name);
  std::map<std::shared_ptr<sem::LoadExpr>, std::string> GetGlobalLoadExprMap(const sem::ExprPtr& p);

  void SingleElementWrite(const sem::LValPtr& lhs, const sem::ExprPtr& rhs);
  void assign_global_var_to_temp(const sem::ExprPtr& e);

  std::string GetLValueName(const sem::LValPtr& lv);
  std::map<std::string, int> vector_stride_map;
  std::set<std::string> element_offset_vector;

  bool depend_on_local_id(const sem::ExprPtr& init);

  lang::Scope<sem::Type>* scope_;
  bool is_sub_group_broadcast_first_val = false;
  bool in_read_statement = false;
  bool in_write_statement = false;
  bool in_cmamp_function = false;
  sem::Type write_type;
  bool in_declare_stmt = false;
  int temp_var_num = 0;
  std::map<std::string, std::string> input_replace_map;
  std::set<std::string> dependent_index;
  std::set<std::string> large_sparse_vactor;
  lang::KernelInfo ki_;
};

static std::map<std::pair<DataType, sem::LimitConst::Which>, std::string> LimitConstLookup = {
    {{DataType::BOOLEAN, sem::LimitConst::MIN}, "0"},        {{DataType::BOOLEAN, sem::LimitConst::MAX}, "0"},
    {{DataType::INT8, sem::LimitConst::MIN}, "SCHAR_MIN"},   {{DataType::INT8, sem::LimitConst::MAX}, "SCHAR_MAX"},
    {{DataType::INT16, sem::LimitConst::MIN}, "SHRT_MIN"},   {{DataType::INT16, sem::LimitConst::MAX}, "SHRT_MAX"},
    {{DataType::INT32, sem::LimitConst::MIN}, "INT_MIN"},    {{DataType::INT32, sem::LimitConst::MAX}, "INT_MAX"},
    {{DataType::INT64, sem::LimitConst::MIN}, "LONG_MIN"},   {{DataType::INT64, sem::LimitConst::MAX}, "LONG_MAX"},
    {{DataType::UINT8, sem::LimitConst::MIN}, "0"},          {{DataType::UINT8, sem::LimitConst::MAX}, "UCHAR_MAX"},
    {{DataType::UINT16, sem::LimitConst::MIN}, "0"},         {{DataType::UINT16, sem::LimitConst::MAX}, "USHRT_MAX"},
    {{DataType::UINT32, sem::LimitConst::MIN}, "0"},         {{DataType::UINT32, sem::LimitConst::MAX}, "UINT_MAX"},
    {{DataType::UINT64, sem::LimitConst::MIN}, "0"},         {{DataType::UINT64, sem::LimitConst::MAX}, "ULONG_MAX"},
    {{DataType::FLOAT32, sem::LimitConst::MIN}, "-FLT_MAX"}, {{DataType::FLOAT32, sem::LimitConst::MAX}, "FLT_MAX"},
    {{DataType::FLOAT64, sem::LimitConst::MIN}, "-DBL_MAX"}, {{DataType::FLOAT64, sem::LimitConst::MAX}, "DBL_MAX"},
};

static std::map<std::string, std::string> FuncNameMap = {
    {"floor", "_FLOOR"}, {"ceil", "_CEIL"}, {"exp", "_EXP"}, {"log", "_LOG"},   {"sqrt", "_SQRT"},
    {"pow", "_POW"},     {"sin", "_SIN"},   {"cos", "_COS"}, {"tanh", "_TANH"}, {"round", "_ROUND"}};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
