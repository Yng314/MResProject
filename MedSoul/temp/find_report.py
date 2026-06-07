import zipfile
import os

def find_report_by_study_id(study_id):
    """
    根据study_id在mimic-cxr-reports.zip中查找并打印对应的报告信息
    报告存储在txt文件中，路径格式为: files/p{subject_id}/p{subject_id}/s{study_id}.txt
    """
    zip_path = "datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-reports.zip"
    
    if not os.path.exists(zip_path):
        print(f"错误: 找不到文件 {zip_path}")
        return
    
    try:
        # 打开zip文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 列出zip文件中的所有文件
            file_list = zip_ref.namelist()
            
            # 查找匹配study_id的txt文件
            # 文件名格式: s{study_id}.txt
            target_filename = f"s{study_id}.txt"
            
            matching_files = []
            for file_path in file_list:
                if file_path.endswith(target_filename):
                    matching_files.append(file_path)
            
            if len(matching_files) == 0:
                print(f"未找到 study_id={study_id} 的报告文件")
                print(f"查找的文件名: {target_filename}")
                return
            
            # 打印找到的文件
            print("="*80)
            print(f"找到 {len(matching_files)} 个匹配的报告:")
            print("="*80)
            
            for file_path in matching_files:
                # 从路径中提取subject_id
                # 路径格式: files/p{subject_id}/p{subject_id}/s{study_id}.txt
                parts = file_path.split('/')
                if len(parts) >= 3:
                    subject_id = parts[1][1:]  # 去掉'p'前缀
                else:
                    subject_id = "Unknown"
                
                print(f"\n文件路径: {file_path}")
                print(f"Subject ID (Project ID): {subject_id}")
                print(f"Study ID: {study_id}")
                print(f"\nReport内容:")
                print("-"*80)
                
                # 读取txt文件内容
                with zip_ref.open(file_path) as f:
                    report_text = f.read().decode('utf-8')
                    print(report_text)
                
                print("-"*80)
                
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 获取用户输入
    study_id = input("请输入 study_id (例如: 50563564): ").strip()
    
    if study_id:
        find_report_by_study_id(study_id)
    else:
        print("错误: study_id 不能为空")

