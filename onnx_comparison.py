#!/usr/bin/env python3
"""
詳細的 ONNX 檔案比較工具
比較舊 ONNX 和新 ONNX 的結構差異
"""

import onnx
import numpy as np
import onnxruntime as ort
from collections import defaultdict, Counter
import json
from typing import Dict, List, Tuple, Any


def analyze_onnx_structure(onnx_path: str) -> Dict[str, Any]:
    """分析 ONNX 檔案的結構"""
    model = onnx.load(onnx_path)
    
    # 基本資訊
    info = {
        'path': onnx_path,
        'total_nodes': len(model.graph.node),
        'total_parameters': len(model.graph.initializer),
        'inputs': [(inp.name, [d.dim_value if d.dim_value > 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]) 
                   for inp in model.graph.input],
        'outputs': [(out.name, [d.dim_value if d.dim_value > 0 else 'dynamic' for d in out.type.tensor_type.shape.dim]) 
                    for out in model.graph.output],
    }
    
    # 節點統計
    node_types = Counter(node.op_type for node in model.graph.node)
    info['node_types'] = dict(node_types)
    
    # 詳細節點資訊
    nodes_info = []
    for i, node in enumerate(model.graph.node):
        node_info = {
            'index': i,
            'name': node.name,
            'op_type': node.op_type,
            'inputs': list(node.input),
            'outputs': list(node.output),
            'attributes': {attr.name: attr for attr in node.attribute}
        }
        nodes_info.append(node_info)
    
    info['nodes'] = nodes_info
    
    # 最後幾個節點（通常是最重要的）
    info['last_nodes'] = nodes_info[-10:] if len(nodes_info) >= 10 else nodes_info
    
    return info


def compare_onnx_files(old_path: str, new_path: str) -> Dict[str, Any]:
    """比較兩個 ONNX 檔案"""
    print(f"=== 載入 ONNX 檔案 ===")
    old_info = analyze_onnx_structure(old_path)
    new_info = analyze_onnx_structure(new_path)
    
    print(f"舊 ONNX: {old_path}")
    print(f"新 ONNX: {new_path}")
    
    comparison = {
        'old': old_info,
        'new': new_info,
        'differences': {}
    }
    
    # 基本統計比較
    print(f"\n=== 基本統計比較 ===")
    print(f"節點總數:")
    print(f"  舊: {old_info['total_nodes']}")
    print(f"  新: {new_info['total_nodes']}")
    print(f"  差異: {new_info['total_nodes'] - old_info['total_nodes']}")
    
    print(f"\n參數總數:")
    print(f"  舊: {old_info['total_parameters']}")
    print(f"  新: {new_info['total_parameters']}")
    print(f"  差異: {new_info['total_parameters'] - old_info['total_parameters']}")
    
    # 輸入輸出比較
    print(f"\n=== 輸入輸出比較 ===")
    print(f"輸入:")
    print(f"  舊: {old_info['inputs']}")
    print(f"  新: {new_info['inputs']}")
    
    print(f"\n輸出:")
    print(f"  舊: {old_info['outputs']}")
    print(f"  新: {new_info['outputs']}")
    
    # 節點類型比較
    print(f"\n=== 節點類型比較 ===")
    old_types = set(old_info['node_types'].keys())
    new_types = set(new_info['node_types'].keys())
    
    common_types = old_types & new_types
    only_old = old_types - new_types
    only_new = new_types - old_types
    
    print(f"共同節點類型: {sorted(common_types)}")
    if only_old:
        print(f"僅在舊 ONNX 中: {sorted(only_old)}")
    if only_new:
        print(f"僅在新 ONNX 中: {sorted(only_new)}")
    
    # 詳細節點類型統計
    print(f"\n=== 詳細節點類型統計 ===")
    all_types = old_types | new_types
    for op_type in sorted(all_types):
        old_count = old_info['node_types'].get(op_type, 0)
        new_count = new_info['node_types'].get(op_type, 0)
        diff = new_count - old_count
        print(f"  {op_type}: 舊={old_count}, 新={new_count}, 差異={diff:+d}")
    
    # 最後節點比較
    print(f"\n=== 最後節點比較 ===")
    print(f"舊 ONNX 最後 10 個節點:")
    for node in old_info['last_nodes']:
        print(f"  {node['index']}: {node['op_type']} - {node['name']}")
    
    print(f"\n新 ONNX 最後 10 個節點:")
    for node in new_info['last_nodes']:
        print(f"  {node['index']}: {node['op_type']} - {node['name']}")
    
    # 關鍵差異分析
    print(f"\n=== 關鍵差異分析 ===")
    
    # Shape/Slice 節點分析
    old_shape_count = old_info['node_types'].get('Shape', 0)
    old_slice_count = old_info['node_types'].get('Slice', 0)
    old_constant_count = old_info['node_types'].get('Constant', 0)
    
    new_shape_count = new_info['node_types'].get('Shape', 0)
    new_slice_count = new_info['node_types'].get('Slice', 0)
    new_constant_count = new_info['node_types'].get('Constant', 0)
    
    print(f"Shape 節點:")
    print(f"  舊: {old_shape_count}, 新: {new_shape_count}, 差異: {new_shape_count - old_shape_count}")
    
    print(f"Slice 節點:")
    print(f"  舊: {old_slice_count}, 新: {new_slice_count}, 差異: {new_slice_count - old_slice_count}")
    
    print(f"Constant 節點:")
    print(f"  舊: {old_constant_count}, 新: {new_constant_count}, 差異: {new_constant_count - old_constant_count}")
    
    # Sigmoid 節點分析
    old_sigmoid_count = old_info['node_types'].get('Sigmoid', 0)
    new_sigmoid_count = new_info['node_types'].get('Sigmoid', 0)
    print(f"Sigmoid 節點:")
    print(f"  舊: {old_sigmoid_count}, 新: {new_sigmoid_count}, 差異: {new_sigmoid_count - old_sigmoid_count}")
    
    # Concat 節點分析
    old_concat_count = old_info['node_types'].get('Concat', 0)
    new_concat_count = new_info['node_types'].get('Concat', 0)
    print(f"Concat 節點:")
    print(f"  舊: {old_concat_count}, 新: {new_concat_count}, 差異: {new_concat_count - old_concat_count}")
    
    # Reshape 節點分析
    old_reshape_count = old_info['node_types'].get('Reshape', 0)
    new_reshape_count = new_info['node_types'].get('Reshape', 0)
    print(f"Reshape 節點:")
    print(f"  舊: {old_reshape_count}, 新: {new_reshape_count}, 差異: {new_reshape_count - old_reshape_count}")
    
    # Transpose 節點分析
    old_transpose_count = old_info['node_types'].get('Transpose', 0)
    new_transpose_count = new_info['node_types'].get('Transpose', 0)
    print(f"Transpose 節點:")
    print(f"  舊: {old_transpose_count}, 新: {new_transpose_count}, 差異: {new_transpose_count - old_transpose_count}")
    
    # 測試推理比較
    print(f"\n=== 推理測試比較 ===")
    try:
        # 載入兩個模型
        old_session = ort.InferenceSession(old_path)
        new_session = ort.InferenceSession(new_path)
        
        # 準備測試輸入
        input_name = old_session.get_inputs()[0].name
        test_input = np.random.randn(1, 3, 960, 960).astype(np.float32)
        
        # 舊模型推理
        old_outputs = old_session.run(None, {input_name: test_input})
        print(f"舊模型輸出:")
        for i, output in enumerate(old_outputs):
            print(f"  輸出 {i}: shape={output.shape}, dtype={output.dtype}")
        
        # 新模型推理
        new_outputs = new_session.run(None, {input_name: test_input})
        print(f"新模型輸出:")
        for i, output in enumerate(new_outputs):
            print(f"  輸出 {i}: shape={output.shape}, dtype={output.dtype}")
        
        # 比較輸出
        if len(old_outputs) == len(new_outputs):
            print(f"\n輸出比較:")
            for i, (old_out, new_out) in enumerate(zip(old_outputs, new_outputs)):
                if old_out.shape == new_out.shape:
                    diff = np.abs(old_out - new_out)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    print(f"  輸出 {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
                else:
                    print(f"  輸出 {i}: 形狀不匹配 - 舊:{old_out.shape}, 新:{new_out.shape}")
        else:
            print(f"輸出數量不匹配: 舊={len(old_outputs)}, 新={len(new_outputs)}")
            
    except Exception as e:
        print(f"推理測試失敗: {e}")
    
    # 總結
    print(f"\n=== 總結 ===")
    total_diff = new_info['total_nodes'] - old_info['total_nodes']
    shape_diff = new_shape_count - old_shape_count
    slice_diff = new_slice_count - old_slice_count
    
    if shape_diff == 0 and slice_diff == 0:
        print("✅ Shape 和 Slice 節點數量一致")
    else:
        print(f"❌ Shape 和 Slice 節點有差異: Shape={shape_diff:+d}, Slice={slice_diff:+d}")
    
    if total_diff == 0:
        print("✅ 總節點數一致")
    else:
        print(f"⚠️  總節點數有差異: {total_diff:+d}")
    
    if old_info['outputs'] == new_info['outputs']:
        print("✅ 輸出格式一致")
    else:
        print("❌ 輸出格式不一致")
    
    return comparison


def main():
    """主函數"""
    old_onnx = "/workspace/work_dirs/old_yolox_elan/yolox_s_opt_elan_batch_6.onnx"
    new_onnx = "/workspace/work_dirs/yolox_opt_elan_deployment_comparison/yolox_opt_elan.onnx"
    
    print("ONNX 檔案詳細比較工具")
    print("=" * 50)
    
    comparison = compare_onnx_files(old_onnx, new_onnx)
    
    # 保存比較結果
    with open("/workspace/onnx_comparison_result.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print(f"\n比較結果已保存到: /workspace/onnx_comparison_result.json")


if __name__ == "__main__":
    main()
