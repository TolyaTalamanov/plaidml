
#include "tile/hal/cm/opt.h"

#include "tile/lang/exprtype.h"
#include "tile/lang/sembuilder.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

namespace {

// This is an instruction optimizer. It's intended to replace expression
// patterns with potentially
// more efficient cm builtin functions.
class InsnOptimizer : public sem::Visitor {
 public:
  explicit InsnOptimizer(bool cl_khr_fp16, const hal::proto::HardwareSettings& settings)
      : cl_khr_fp16_{cl_khr_fp16}, settings_{settings} {}

  void Visit(const sem::IntConst& node) override {}

  void Visit(const sem::FloatConst& node) override {}

  void Visit(const sem::LookupLVal& node) override {}

  void Visit(const sem::LoadExpr& node) override {
    if (node.inner) node.inner->Accept(*this);
  }

  void Visit(const sem::StoreStmt& node) override {
    if (node.lhs) node.lhs->Accept(*this);
    if (node.rhs) node.rhs->Accept(*this);
  }

  void Visit(const sem::SubscriptLVal& node) override {
    if (node.ptr) node.ptr->Accept(*this);
    if (node.offset) node.offset->Accept(*this);
  }

  void Visit(const sem::DeclareStmt& node) override {
    /*
    Type type;
    std::string name;
    ExprPtr init;
    */
    if (node.init) node.init->Accept(*this);
  }

  void Visit(const sem::UnaryExpr& node) override {
    /*
    std::string op;
    ExprPtr inner;
    */
    if (node.inner) node.inner->Accept(*this);
  }

  void Visit(const sem::BinaryExpr& node) override {
    /*
    std::string op;
    ExprPtr lhs;
    ExprPtr rhs;
    */
    if (node.lhs) node.lhs->Accept(*this);
    if (node.rhs) node.rhs->Accept(*this);
  }

  void Visit(const sem::CondExpr& node) override {
    if (node.cond) node.cond->Accept(*this);
    if (node.tcase) node.tcase->Accept(*this);
    if (node.fcase) node.fcase->Accept(*this);
  }

  void Visit(const sem::SelectExpr& node) override {
    if (node.cond) node.cond->Accept(*this);
    if (node.tcase) node.tcase->Accept(*this);
    if (node.fcase) node.fcase->Accept(*this);
  }

  void Visit(const sem::ClampExpr& node) override {
    if (node.val) node.val->Accept(*this);
    if (node.min) node.min->Accept(*this);
    if (node.max) node.max->Accept(*this);
  }

  void Visit(const sem::CastExpr& node) override {
    /*
    Type type;
    ExprPtr val;
    */
    if (node.val) node.val->Accept(*this);
  }

  void Visit(const sem::CallExpr& node) override {
    /*
  enum class Function {
    ACOS,
    ASIN,
    ATAN,
    CEIL,
    COS,
    COSH,
    EXP,
    FLOOR,
    LOG,
    MAD,
    POW,
    ROUND,
    SIN,
    SINH,
    SQRT,
    SUB_GROUP_BROADCAST,
    TAN,
    TANH,
  };
  Function function;
  std::string name;
  std::vector<ExprPtr> vals;
   */
    for (size_t i = 0; i < node.vals.size(); i++) {
      if (node.vals[i]) node.vals[i]->Accept(*this);
    }
  }

  void Visit(const sem::LimitConst& node) override {}

  void Visit(const sem::IndexExpr& node) override {}

  void Visit(const sem::Block& node) override {
    for (const sem::StmtPtr& ptr : node.statements) {
      if (ptr) ptr->Accept(*this);
    }
  }

  void Visit(const sem::IfStmt& node) override {
    if (node.cond) node.cond->Accept(*this);
    if (node.iftrue) node.iftrue->Accept(*this);
    if (node.iffalse) node.iffalse->Accept(*this);
  }

  void Visit(const sem::ForStmt& node) override {
    if (node.inner) node.inner->Accept(*this);
  }

  void Visit(const sem::WhileStmt& node) override {
    if (node.cond) node.cond->Accept(*this);
    if (node.inner) node.inner->Accept(*this);
  }

  void Visit(const sem::BarrierStmt& node) override {}

  void Visit(const sem::ReturnStmt& node) override {
    if (node.value) node.value->Accept(*this);
  }

  void Visit(const sem::SpecialStmt& node) override {
    for (size_t i = 0; i < node.params.size(); i++) {
      if (node.params[i]) node.params[i]->Accept(*this);
    }
  }

  void Visit(const sem::Function& node) override {
    if (node.body) node.body->Accept(*this);
  }

 private:
  bool cl_khr_fp16_;
  hal::proto::HardwareSettings settings_;
};

}  // namespace

void OptimizeKernel(const lang::KernelInfo& ki, bool cl_khr_fp16, const hal::proto::HardwareSettings& settings) {
  InsnOptimizer opt(cl_khr_fp16, settings);
  ki.kfunc->Accept(opt);
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
