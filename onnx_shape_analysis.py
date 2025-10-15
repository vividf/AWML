#!/usr/bin/env python3
"""
ONNX 差異深度分析
分析為什麼新 ONNX 仍然有 Shape 節點
"""

import onnx
import numpy as np
from collections import defaultdict


def analyze_shape_nodes(onnx_path: str):
    """分析 Shape 節點的來源"""
    model = onnx.load(onnx_path)
    
    print(f"=== 分析 {onnx_path} 中的 Shape 節點 ===")
    
    # 找到所有 Shape 節點
    shape_nodes = [node for node in model.graph.node if node.op_type == 'Shape']
    print(f"找到 {len(shape_nodes)} 個 Shape 節點:")
    
    for i, node in enumerate(shape_nodes):
        print(f"\nShape 節點 {i+1}: {node.name}")
        print(f"  輸入: {list(node.input)}")
        print(f"  輸出: {list(node.output)}")
        
        # 找到使用這個 Shape 輸出的節點
        consumers = []
        for other_node in model.graph.node:
            if any(output in other_node.input for output in node.output):
                consumers.append(other_node)
        
        print(f"  使用此輸出的節點:")
        for consumer in consumers:
            print(f"    {consumer.op_type} - {consumer.name}")
            print(f"      輸入: {list(consumer.input)}")
    
    # 找到所有 Unsqueeze 節點
    unsqueeze_nodes = [node for node in model.graph.node if node.op_type == 'Unsqueeze']
    print(f"\n找到 {len(unsqueeze_nodes)} 個 Unsqueeze 節點:")
    
    for i, node in enumerate(unsqueeze_nodes):
        print(f"\nUnsqueeze 節點 {i+1}: {node.name}")
        print(f"  輸入: {list(node.input)}")
        print(f"  輸出: {list(node.output)}")
        
        # 檢查 attributes
        for attr in node.attribute:
            print(f"  屬性 {attr.name}: {attr}")
    
    # 找到所有 Gather 節點
    gather_nodes = [node for node in model.graph.node if node.op_type == 'Gather']
    print(f"\n找到 {len(gather_nodes)} 個 Gather 節點:")
    
    for i, node in enumerate(gather_nodes):
        print(f"\nGather 節點 {i+1}: {node.name}")
        print(f"  輸入: {list(node.input)}")
        print(f"  輸出: {list(node.output)}")
    
    # 分析最後的節點序列
    print(f"\n=== 最後節點序列分析 ===")
    last_nodes = model.graph.node[-15:]  # 最後 15 個節點
    
    for i, node in enumerate(last_nodes):
        print(f"{len(model.graph.node)-15+i:3d}: {node.op_type:10s} - {node.name}")
    
    return shape_nodes, unsqueeze_nodes, gather_nodes


def analyze_reshape_operations(onnx_path: str):
    """分析 Reshape 操作的詳細資訊"""
    model = onnx.load(onnx_path)
    
    print(f"\n=== 分析 {onnx_path} 中的 Reshape 操作 ===")
    
    reshape_nodes = [node for node in model.graph.node if node.op_type == 'Reshape']
    print(f"找到 {len(reshape_nodes)} 個 Reshape 節點:")
    
    for i, node in enumerate(reshape_nodes):
        print(f"\nReshape 節點 {i+1}: {node.name}")
        print(f"  輸入: {list(node.input)}")
        print(f"  輸出: {list(node.output)}")
        
        # 檢查是否有 shape 輸入
        if len(node.input) >= 2:
            shape_input = node.input[1]
            print(f"  Shape 輸入: {shape_input}")
            
            # 找到提供 shape 的節點
            shape_provider = None
            for other_node in model.graph.node:
                if shape_input in other_node.output:
                    shape_provider = other_node
                    break
            
            if shape_provider:
                print(f"  Shape 提供者: {shape_provider.op_type} - {shape_provider.name}")
                if shape_provider.op_type == 'Shape':
                    print(f"    ⚠️  這個 Reshape 使用了 Shape 節點的輸出！")
    
    return reshape_nodes


def compare_reshape_patterns(old_path: str, new_path: str):
    """比較兩個 ONNX 的 Reshape 模式"""
    print(f"\n=== Reshape 模式比較 ===")
    
    old_reshapes = analyze_reshape_operations(old_path)
    new_reshapes = analyze_reshape_operations(new_path)
    
    print(f"\n舊 ONNX Reshape 節點數量: {len(old_reshapes)}")
    print(f"新 ONNX Reshape 節點數量: {len(new_reshapes)}")
    
    # 檢查是否有 Reshape 使用了 Shape 輸出
    old_model = onnx.load(old_path)
    new_model = onnx.load(new_path)
    
    old_shape_dependent = 0
    new_shape_dependent = 0
    
    for node in old_model.graph.node:
        if node.op_type == 'Reshape' and len(node.input) >= 2:
            shape_input = node.input[1]
            for other_node in old_model.graph.node:
                if shape_input in other_node.output and other_node.op_type == 'Shape':
                    old_shape_dependent += 1
                    break
    
    for node in new_model.graph.node:
        if node.op_type == 'Reshape' and len(node.input) >= 2:
            shape_input = node.input[1]
            for other_node in new_model.graph.node:
                if shape_input in other_node.output and other_node.op_type == 'Shape':
                    new_shape_dependent += 1
                    break
    
    print(f"\n依賴 Shape 節點的 Reshape 數量:")
    print(f"  舊: {old_shape_dependent}")
    print(f"  新: {new_shape_dependent}")
    
    return old_shape_dependent, new_shape_dependent


def main():
    """主函數"""
    old_onnx = "/workspace/work_dirs/old_yolox_elan/yolox_s_opt_elan_batch_6.onnx"
    new_onnx = "/workspace/work_dirs/yolox_opt_elan_deployment_comparison/yolox_opt_elan.onnx"
    
    print("ONNX Shape 節點深度分析")
    print("=" * 50)
    
    # 分析新 ONNX 的 Shape 節點
    shape_nodes, unsqueeze_nodes, gather_nodes = analyze_shape_nodes(new_onnx)
    
    # 分析舊 ONNX（應該沒有 Shape 節點）
    print(f"\n" + "="*50)
    old_model = onnx.load(old_onnx)
    old_shape_nodes = [node for node in old_model.graph.node if node.op_type == 'Shape']
    print(f"舊 ONNX 中的 Shape 節點數量: {len(old_shape_nodes)}")
    
    # 比較 Reshape 模式
    compare_reshape_patterns(old_onnx, new_onnx)
    
    # 總結分析
    print(f"\n=== 差異原因分析 ===")
    print(f"1. 新 ONNX 有 {len(shape_nodes)} 個 Shape 節點，舊 ONNX 有 {len(old_shape_nodes)} 個")
    print(f"2. 新 ONNX 有 {len(unsqueeze_nodes)} 個 Unsqueeze 節點")
    print(f"3. 新 ONNX 有 {len(gather_nodes)} 個 Gather 節點")
    print(f"4. 這些節點通常用於動態 Reshape 操作")
    
    print(f"\n=== 可能的解決方案 ===")
    print(f"1. 檢查 onnx_wrapper.py 中的 reshape 操作")
    print(f"2. 確保使用靜態 shape 而不是動態 shape")
    print(f"3. 檢查是否有其他操作導致動態 shape")
    print(f"4. 考慮使用固定 batch size 而不是動態 batch size")


if __name__ == "__main__":
    main()
