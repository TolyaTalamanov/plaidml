// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <string>
#include <vector>

#include "tile/base/hal.h"
#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

class TravelVisitor : public sem::Visitor {
 public:
  enum TravelType { GET_STRING };

  void Visit(const sem::IntConst& node) override;
  void Visit(const sem::FloatConst& node) override;
  void Visit(const sem::LookupLVal& node) override;
  void Visit(const sem::LoadExpr& node) override;
  void Visit(const sem::StoreStmt& node) override;
  void Visit(const sem::SubscriptLVal& node) override;
  void Visit(const sem::DeclareStmt& node) override;
  void Visit(const sem::UnaryExpr& node) override;
  void Visit(const sem::BinaryExpr& node) override;
  void Visit(const sem::CondExpr& node) override;
  void Visit(const sem::SelectExpr& node) override;
  void Visit(const sem::ClampExpr& node) override;
  void Visit(const sem::CastExpr& node) override;
  void Visit(const sem::CallExpr& node) override;
  void Visit(const sem::LimitConst& node) override;
  void Visit(const sem::IndexExpr& node) override;
  void Visit(const sem::Block& node) override;
  void Visit(const sem::IfStmt& node) override;
  void Visit(const sem::ForStmt& node) override;
  void Visit(const sem::WhileStmt& node) override;
  void Visit(const sem::BarrierStmt& node) override;
  void Visit(const sem::ReturnStmt& node) override;
  void Visit(const sem::SpecialStmt& node) override;
  void Visit(const sem::Function& node) override;

  void init_node_str() {
    travel = GET_STRING;
    node_str.str("");
  }
  std::string get_node_str() const { return node_str.str(); }

 private:
  TravelType travel;
  std::ostringstream node_str;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
