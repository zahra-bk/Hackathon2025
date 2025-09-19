def process_data(data_file):
    with open(data_file, 'r') as f:
        content = json.load(f)
    
    results = []
    for item in content:
        if item['status'] == 'pending':
            results.append(item['id'])
    
    return results