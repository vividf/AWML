import mmengine

# load the info.pkl file
data = mmengine.load("/workspace/data/t4dataset/calibration_info/t4dataset_x2_calib_infos_test.pkl")

print(f"Type: {type(data)}")

if isinstance(data, list):
    print(f"Direct list with {len(data)} samples")
    print(f"First sample keys: {list(data[0].keys())}")

elif isinstance(data, dict):
    print(f"Dict with keys: {list(data.keys())}")
    if "data_list" in data:
        print(f"data_list contains {len(data['data_list'])} samples")
        print(f"First sample keys: {list(data['data_list'][0].keys())}")
    if "metainfo" in data:
        print(f"metainfo keys: {list(data['metainfo'].keys())}")
