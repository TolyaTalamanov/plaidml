
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

  void Visit(const sem::LoadExpr& node) override {}

  void Visit(const sem::StoreStmt& node) override {}

  void Visit(const sem::SubscriptLVal& node) override {}

  void Visit(const sem::DeclareStmt& node) override {}

  void Visit(const sem::UnaryExpr& node) override {}

  void Visit(const sem::BinaryExpr& node) override {}

  void Visit(const sem::CondExpr& node) override {}

  void Visit(const sem::SelectExpr& node) override {}

  void Visit(const sem::ClampExpr& node) override {}

  void Visit(const sem::CastExpr& node) override {}

  void Visit(const sem::CallExpr& node) override {}

  void Visit(const sem::LimitConst& node) override {}

  void Visit(const sem::IndexExpr& node) override {}

  void Visit(const sem::Block& node) override {}

  void Visit(const sem::IfStmt& node) override {}

  void Visit(const sem::ForStmt& node) override {}

  void Visit(const sem::WhileStmt& node) override {}

  void Visit(const sem::BarrierStmt& node) override {}

  void Visit(const sem::ReturnStmt& node) override {}

  void Visit(const sem::SpecialStmt& node) override {}

  void Visit(const sem::Function& node) override {}

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
