// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/travel.h"
#include "tile/hal/cm/emitcm.h"

#include "tile/lang/exprtype.h"
#include "tile/lang/sembuilder.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

void TravelVisitor::Visit(const sem::IntConst& node) {
  if (travel == GET_STRING) {
    node_str << std::to_string(node.value);
    return;
  }
}

void TravelVisitor::Visit(const sem::FloatConst& node) {
  if (travel == GET_STRING) {
    node_str << std::to_string(node.value);
    return;
  }
}

void TravelVisitor::Visit(const sem::LookupLVal& node) {
  if (travel == GET_STRING) {
    node_str << node.name;
    return;
  }
  if (travel == GET_GLOBAL_VAR_WITH_OFFSET) {
    if (global_params.find(node.name) != global_params.end()) {
      global_var_with_offset << node.name;
    }
    return;
  }
}

void TravelVisitor::Visit(const sem::SubscriptLVal& node) {
  if (travel == GET_STRING) {
    node_str << "(";
    if (node.ptr) node.ptr->Accept(*this);
    node_str << " [";
    if (node.offset) node.offset->Accept(*this);
    node_str << "])";
    return;
  }
  if (travel == GET_GLOBAL_VAR_WITH_OFFSET) {
    node.ptr->Accept(*this);

    auto s = GetGlobalVarWithOffset();
    if (s.size() > 0) {
      InitNodeStr();
      node.offset->Accept(*this);
      auto node_str = GetNodeStr();
      global_var_with_offset << " " << node_str;

      travel = GET_GLOBAL_VAR_WITH_OFFSET;
    }
    return;
  }
}

void TravelVisitor::Visit(const sem::LoadExpr& node) {
  if (travel == GET_STRING) {
    if (node.inner) node.inner->Accept(*this);
    return;
  }

  if (travel == GET_GLOBAL_LOAD_EXPRS) {
    InitGlobalVarWithOffset();
    if (node.inner) node.inner->Accept(*this);
    auto s = GetGlobalVarWithOffset();

    travel = GET_GLOBAL_LOAD_EXPRS;

    if (s.length() > 0) {
      global_load_exprs[std::make_shared<sem::LoadExpr>(node)] = s;
    }
    return;
  }
}

void TravelVisitor::Visit(const sem::StoreStmt& node) {
  if (travel == GET_STRING) {
    if (node.lhs) node.lhs->Accept(*this);
    node_str << " = ";
    if (node.rhs) node.rhs->Accept(*this);
    return;
  }
}

void TravelVisitor::Visit(const sem::DeclareStmt& node) {
  if (travel == GET_STRING) {
    node_str << node.name;
    node_str << " = ";
    if (node.init) node.init->Accept(*this);
    return;
  }
}

void TravelVisitor::Visit(const sem::UnaryExpr& node) {
  if (travel == GET_STRING) {
    node_str << node.op;
    node_str << " : ";
    if (node.inner) node.inner->Accept(*this);
    return;
  }
}

void TravelVisitor::Visit(const sem::BinaryExpr& node) {
  if (travel == GET_STRING) {
    node_str << "(";
    if (node.lhs) node.lhs->Accept(*this);
    node_str << " " << node.op << " ";
    if (node.rhs) node.rhs->Accept(*this);
    node_str << ")";
    return;
  }

  if (travel == GET_GLOBAL_LOAD_EXPRS) {
    if (node.lhs) node.lhs->Accept(*this);
    if (node.rhs) node.rhs->Accept(*this);
    return;
  }
}

void TravelVisitor::Visit(const sem::CondExpr& node) {
  if (travel == GET_STRING) {
    node_str << "(";
    if (node.cond) node.cond->Accept(*this);
    node_str << " ";
    if (node.tcase) node.tcase->Accept(*this);
    node_str << " ";
    if (node.fcase) node.fcase->Accept(*this);
    node_str << ")";
    return;
  }
  if (travel == GET_GLOBAL_LOAD_EXPRS) {
    if (node.cond) node.cond->Accept(*this);
    if (node.tcase) node.tcase->Accept(*this);
    if (node.fcase) node.fcase->Accept(*this);
    return;
  }
}

void TravelVisitor::Visit(const sem::SelectExpr& node) {
  if (travel == GET_STRING) {
    node_str << "(";
    if (node.cond) node.cond->Accept(*this);
    node_str << " ";
    if (node.tcase) node.tcase->Accept(*this);
    node_str << " ";
    if (node.fcase) node.fcase->Accept(*this);
    node_str << ")";
    return;
  }
  if (travel == GET_GLOBAL_LOAD_EXPRS) {
    if (node.cond) node.cond->Accept(*this);
    if (node.tcase) node.tcase->Accept(*this);
    if (node.fcase) node.fcase->Accept(*this);
    return;
  }
}

void TravelVisitor::Visit(const sem::ClampExpr& node) {
  if (travel == GET_STRING) {
    node_str << "(";
    if (node.val) node.val->Accept(*this);
    node_str << " ";
    if (node.min) node.min->Accept(*this);
    node_str << " ";
    if (node.max) node.max->Accept(*this);
    node_str << ")";
    return;
  }
  if (travel == GET_GLOBAL_LOAD_EXPRS) {
    if (node.val) node.val->Accept(*this);
    if (node.min) node.min->Accept(*this);
    if (node.max) node.max->Accept(*this);
    return;
  }
}

void TravelVisitor::Visit(const sem::CastExpr& node) {
  if (travel == GET_STRING) {
    if (node.val) node.val->Accept(*this);
    return;
  }
  if (travel == GET_GLOBAL_LOAD_EXPRS) {
    if (node.val) node.val->Accept(*this);
    return;
  }
}

void TravelVisitor::Visit(const sem::CallExpr& node) {
  if (travel == GET_STRING) {
    node_str << node.name << "(";
    for (size_t i = 0; i < node.vals.size(); i++) {
      if (node.vals[i]) node.vals[i]->Accept(*this);
      if (i < node.vals.size() - 1) {
        node_str << ", ";
      }
    }
    node_str << ")";
    return;
  }
  if (travel == GET_GLOBAL_LOAD_EXPRS) {
    for (size_t i = 0; i < node.vals.size(); i++) {
      if (node.vals[i]) node.vals[i]->Accept(*this);
    }
    return;
  }
}

void TravelVisitor::Visit(const sem::LimitConst& node) {
  if (travel == GET_STRING) {
    if (node.which == sem::LimitConst::ZERO) {
      node_str << "0";
    } else if (node.which == sem::LimitConst::ONE) {
      node_str << "1";
    }
    auto it = LimitConstLookup.find(std::make_pair(node.type, node.which));
    if (it == LimitConstLookup.end()) {
      throw std::runtime_error("Invalid type in LimitConst");
    }
    node_str << (it->second);
    return;
  }
}

void TravelVisitor::Visit(const sem::IndexExpr& node) {
  if (travel == GET_STRING) {
    switch (node.type) {
      case sem::IndexExpr::GLOBAL:
        node_str << "global " << std::to_string(node.dim);
        break;
      case sem::IndexExpr::GROUP:
        node_str << "group " << std::to_string(node.dim);
        break;
      case sem::IndexExpr::LOCAL:
        node_str << "local" << std::to_string(node.dim);
        break;
      default:
        node_str << "other index";
    }
    return;
  }
}

void TravelVisitor::Visit(const sem::Block& node) {
  if (travel == GET_STRING) {
    for (const sem::StmtPtr& ptr : node.statements) {
      if (ptr) ptr->Accept(*this);
      node_str << " ";
    }
    return;
  }
}

void TravelVisitor::Visit(const sem::IfStmt& node) {
  if (travel == GET_STRING) {
    node_str << "if(";
    if (node.cond) node.cond->Accept(*this);
    node_str << "; ";
    if (node.iftrue) node.iftrue->Accept(*this);
    node_str << "; ";
    if (node.iffalse) node.iffalse->Accept(*this);
    node_str << ")";
    return;
  }
}

void TravelVisitor::Visit(const sem::ForStmt& node) {
  if (travel == GET_STRING) {
    node_str << "for(";
    if (node.inner) node.inner->Accept(*this);
    node_str << ")";
    return;
  }
}

void TravelVisitor::Visit(const sem::WhileStmt& node) {
  if (travel == GET_STRING) {
    node_str << "while(";
    if (node.cond) node.cond->Accept(*this);
    node_str << "; ";
    if (node.inner) node.inner->Accept(*this);
    node_str << ")";
    return;
  }
}

void TravelVisitor::Visit(const sem::BarrierStmt& node) {
  if (travel == GET_STRING) {
    node_str << "barrier";
    return;
  }
}

void TravelVisitor::Visit(const sem::ReturnStmt& node) {
  if (travel == GET_STRING) {
    node_str << "return ";
    if (node.value) node.value->Accept(*this);
    return;
  }
}

void TravelVisitor::Visit(const sem::SpecialStmt& node) {
  if (travel == GET_STRING) {
    node_str << "special(";
    for (size_t i = 0; i < node.params.size(); i++) {
      if (node.params[i]) node.params[i]->Accept(*this);
      node_str << " ";
    }
    node_str << ")";
    return;
  }
}

void TravelVisitor::Visit(const sem::Function& node) {
  if (travel == GET_STRING) {
    node_str << "function ";
    if (node.body) node.body->Accept(*this);
    return;
  }
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
