//===- MemRefToEmitC.cpp - MemRef to EmitC conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert memref ops into emitc ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct ConvertAlloc final : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    // llvm::errs() << "===========================Converting alloc operation:\n";

    // op.dump();  // Print the original operation
    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    // Ensure that the memref type has a static shape.
    if (!op.getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform alloc with dynamic shape");
    }

    // Ensure the alignment is either unspecified or compatible with EmitC.
    if (op.getAlignment().value_or(1) > 1) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform alloc with alignment requirement");
    }

    // Convert the memref type to EmitC's array type.
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(), "cannot convert type");
    }

    // Replace the op with a new EmitC variable operation.
    auto noInit = emitc::OpaqueAttr::get(getContext(), "");
    Value variableOp = rewriter.create<emitc::VariableOp>(op.getLoc(), resultTy, noInit);
    rewriter.replaceOp(op, {variableOp});

    return success();
  }
};

struct ConvertReturn final : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    // llvm::errs() << "===========================Converting return operation:\n";

    // llvm::errs() << "Before conversion:\n";
    // op.dump();  // Print the original operation
    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    // EmitC return can only handle a single operand.
    if (operands.getOperands().size() == 1) {
      // llvm::errs() << "Replacing return operation with emitc.return operation with 1 operand.\n";
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(op, operands.getOperands()[0]);
    }
    else if (operands.getOperands().size() == 0) {
      // llvm::errs() << "Replacing return operation with emitc.return operation with 0 operands.\n";
      // rewriter.replaceOpWithNewOp<emitc::ReturnOp>(op); // It should be this line, since according
      // to the EmitC dialect, the return operation does support 0 operands. However, if we use this line,
      // the compilation will fail with the following error:
      // error: no matching function for call to ‘mlir::emitc::ReturnOp::build(mlir::OpBuilder&, mlir::OperationState&)’
      // So, we will use the following line instead, which creates an empty emitc.return operation.
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(op, mlir::Value());
    }
    else {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                        "EmitC return can only handle a single operand");
    }

    // llvm::errs() << "After conversion:\n";
    // op->getParentOfType<ModuleOp>().dump();// print the whole module
    
    return success();
  }
};

struct ConvertAlloca final : public OpConversionPattern<memref::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocaOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    // llvm::errs() << "===========================Converting alloca operation:\n";

    if (!op.getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform alloca with dynamic shape");
    }

    if (op.getAlignment().value_or(1) > 1) {
      // TODO: Allow alignment if it is not more than the natural alignment
      // of the C array.
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform alloca with alignment requirement");
    }

    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(), "cannot convert type");
    }
    auto noInit = emitc::OpaqueAttr::get(getContext(), "");
    rewriter.replaceOpWithNewOp<emitc::VariableOp>(op, resultTy, noInit);
    return success();
  }
};

struct ConvertGlobal final : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    // llvm::errs() << "===========================Converting global operation:\n";
    // op.dump();  // Print the original operation

    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    if (!op.getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform global with dynamic shape");
    }

    if (op.getAlignment().value_or(1) > 1) {
      // TODO: Extend GlobalOp to specify alignment via the `alignas` specifier.
      return rewriter.notifyMatchFailure(
          op.getLoc(), "global variable with alignment requirement is "
                       "currently not supported");
    }
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot convert result type");
    }

    SymbolTable::Visibility visibility = SymbolTable::getSymbolVisibility(op);
    if (visibility != SymbolTable::Visibility::Public &&
        visibility != SymbolTable::Visibility::Private) {
      return rewriter.notifyMatchFailure(
          op.getLoc(),
          "only public and private visibility is currently supported");
    }
    // We are explicit in specifing the linkage because the default linkage
    // for constants is different in C and C++.
    bool staticSpecifier = visibility == SymbolTable::Visibility::Private;
    bool externSpecifier = !staticSpecifier;

    Attribute initialValue = operands.getInitialValueAttr();
    if (isa_and_present<UnitAttr>(initialValue))
      initialValue = {};

    rewriter.replaceOpWithNewOp<emitc::GlobalOp>(
        op, operands.getSymName(), resultTy, initialValue, externSpecifier,
        staticSpecifier, operands.getConstant());

    // llvm::errs() << "Replaced global operation with emitc.global operation.\n";

    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    return success();
  }
};

struct ConvertGetGlobal final
    : public OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    // llvm ::errs() << "===========================Converting get_global operation:\n";

    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot convert result type");
    }
    rewriter.replaceOpWithNewOp<emitc::GetGlobalOp>(op, resultTy,
                                                    operands.getNameAttr());
    return success();
  }
};

struct ConvertFuncArguments final : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    // llvm::errs() << "===========================Converting function arguments:\n";

    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    // Convert the function's argument types
    SmallVector<Type, 4> newArgTypes;
    for (Type argType : op.getFunctionType().getInputs()) {
      if (auto memrefType = argType.dyn_cast<MemRefType>()) {
        // Convert memref type to emitc.array type
        auto arrayType = emitc::ArrayType::get(
            memrefType.getShape(), memrefType.getElementType());
        newArgTypes.push_back(arrayType);
      } else {
        newArgTypes.push_back(argType);
      }
    }

    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    // Create the new function type with the converted argument types
    auto newFuncType = FunctionType::get(op.getContext(), newArgTypes,
                                         op.getFunctionType().getResults());

                                         
    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    // Create the converted emitc.func op with the new function type.
    auto newFuncOp = rewriter.create<emitc::FuncOp>(
        op.getLoc(), op.getName(), newFuncType);

        
    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    // Copy over all attributes other than the function name and type.
    for (const auto &namedAttr : op->getAttrs()) {
      if (namedAttr.getName() != op.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName())
        newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    
    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    // Inline the body from the old FuncOp to the new one, if it has a body.
    if (!op.isExternal()) {
      rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                                  newFuncOp.end());

      // Update the block arguments to match the new types
      Block &entryBlock = newFuncOp.getBody().front();
      for (auto [index, newArgType] : llvm::enumerate(newArgTypes)) {
        Value arg = entryBlock.getArgument(index);
        if (arg.getType() != newArgType) {
          arg.setType(newArgType);
        }
      }
    }

    // llvm::errs() << "dumped by op.getParentOfType<ModuleOp>().dump():\n";
    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    // Erase the old FuncOp
    if (op) {
      // llvm::errs() << "Erasing operation:\n";
      // op.dump();
      // op.getOperation()->erase();
      rewriter.eraseOp(op);
    } else {
      llvm::errs() << "Operation already invalid.\n";
    }

    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    // auto funcType = op.getFunctionType(); 
    // llvm::errs() << "Function signature argument type: " << funcType.getInput(0) << "\n";

    // Block &entryBlock = op.getBody().front(); 
    // llvm::errs() << "Entry block argument type: " << entryBlock.getArgument(0).getType() << "\n";


    return success();
  }
};



struct ConvertLoad final : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    // Debug: Print the original operation
    // llvm::errs() << "===========================Converting load operation:\n";
    // op.dump();  // Print the original operation
    
    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    // Convert the result type
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      llvm::errs() << "Failed to convert result type.\n";
      return rewriter.notifyMatchFailure(op.getLoc(), "cannot convert type");
    }

    Value memrefValue = operands.getMemref();

    if (auto unrealizedCast = memrefValue.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (unrealizedCast->getResult(0).getType().isa<emitc::ArrayType>()) {
        memrefValue = unrealizedCast->getResult(0);
        llvm::errs() << "Resolved unrealized conversion cast.\n";
        // Optionally, erase the unrealized cast to clean up the IR
        rewriter.eraseOp(unrealizedCast);
      }
      else {
        llvm::errs() << "Failed to resolve unrealized conversion cast.\n";
        return rewriter.notifyMatchFailure(op.getLoc(), "expected array type");
      }
    }

    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    auto arrayValue =
        dyn_cast<TypedValue<emitc::ArrayType>>(operands.getMemref());
    if (!arrayValue) {
      llvm::errs() << "Expected array type but got:\n";
      memrefValue.getType().dump();  // Print the type that was expected to be an array
      return rewriter.notifyMatchFailure(op.getLoc(), "expected array type");
    }

    // Create the subscript operation
    auto subscript = rewriter.create<emitc::SubscriptOp>(
        op.getLoc(), arrayValue, operands.getIndices());

    // llvm::errs() << "Created emitc.subscript operation:\n";
    // subscript.dump();  // Print the created subscript operation
    
    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    // Create a variable and assign the subscript result to it
    auto noInit = emitc::OpaqueAttr::get(getContext(), "");
    auto var =
        rewriter.create<emitc::VariableOp>(op.getLoc(), resultTy, noInit);

    // llvm::errs() << "Created emitc.variable operation:\n";
    // var.dump();  // Print the created variable operation

    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    rewriter.create<emitc::AssignOp>(op.getLoc(), var, subscript);
    rewriter.replaceOp(op, var);

    // llvm::errs() << "Replaced original operation with the variable.\n";
    
    // op->getParentOfType<ModuleOp>().dump();// print the whole module

    return success();
  }
};

struct ConvertStore final : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    // llvm::errs() << "===========================Converting store operation:\n";
    
    auto arrayValue =
        dyn_cast<TypedValue<emitc::ArrayType>>(operands.getMemref());
    if (!arrayValue) {
      return rewriter.notifyMatchFailure(op.getLoc(), "expected array type");
    }

    auto subscript = rewriter.create<emitc::SubscriptOp>(
        op.getLoc(), arrayValue, operands.getIndices());
    rewriter.replaceOpWithNewOp<emitc::AssignOp>(op, subscript,
                                                 operands.getValue());
    return success();
  }
};
} // namespace

void mlir::populateMemRefToEmitCTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&](MemRefType memRefType) -> std::optional<Type> {
        if (!memRefType.hasStaticShape() ||
            !memRefType.getLayout().isIdentity() || memRefType.getRank() == 0) {
          return {};
        }
        Type convertedElementType =
            typeConverter.convertType(memRefType.getElementType());
        if (!convertedElementType)
          return {};
        return emitc::ArrayType::get(memRefType.getShape(),
                                     convertedElementType);
      });
}

void mlir::populateMemRefToEmitCConversionPatterns(RewritePatternSet &patterns,
                                                   TypeConverter &converter) {
  patterns.add<ConvertAlloc, ConvertReturn, ConvertAlloca, ConvertGlobal, ConvertGetGlobal, ConvertFuncArguments, ConvertLoad,
               ConvertStore>(converter, patterns.getContext());
}
