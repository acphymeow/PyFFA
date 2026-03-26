import os
import threading
import argparse


# 定义线程执行的删除文件函数
def delete_files(file_paths):
    for file_path in file_paths:
        try:
            # 删除文件
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


# 主函数
def main(folder_path, num_threads):

    # --- 第一步：路径安全检查 ---
    current_dir = os.path.abspath(os.getcwd())
    target_dir = os.path.abspath(folder_path)
    common_path = os.path.commonpath([current_dir, target_dir])

    if common_path != current_dir:
        print(f"Error: The folder '{target_dir}' is not within the current directory '{current_dir}'.")
        print("Aborting to avoid unsafe deletion.")
        return
    
    # 可选：如果不允许删除“就是当前目录”的情况，做额外判断
    if target_dir == current_dir:
        print(f"Error: The folder '{target_dir}' is exactly the current directory.")
        print("Aborting to avoid unsafe deletion.")
        return

    # 存储文件夹下所有文件的路径
    all_file_paths = []
    # 递归遍历文件夹，获取所有文件路径
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            all_file_paths.append(os.path.join(root, file))

    # 计算每个线程大致分配到的文件数量
    chunk_size = len(all_file_paths) // num_threads
    threads = []

    # 分配文件路径给每个线程
    for i in range(num_threads):
        start = i * chunk_size
        end = start + chunk_size if i < num_threads - 1 else len(all_file_paths)
        # 提取当前线程要处理的文件路径
        thread_file_paths = all_file_paths[start:end]
        # 创建线程
        thread = threading.Thread(target=delete_files, args=(thread_file_paths,))
        threads.append(thread)
        # 启动线程
        thread.start()

    # 等待所有线程完成任务
    for thread in threads:
        thread.join()
    
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            try:
                os.rmdir(dir_path)  # 仅能删除空文件夹
            except OSError:
                # 如果文件夹不空或其他原因导致删除失败，忽略或打印提示
                pass


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Recursively delete files in a folder using multiple threads.')
    # 添加 -f 选项，指定文件夹路径
    parser.add_argument('-f', '--folder_path', type=str, required=True,
                        help='Path to the folder containing files to be deleted.')
    # 添加 -n 选项，指定线程数量，默认值为 10
    parser.add_argument('-n', '--num_threads', type=int, default=10,
                        help='Number of threads to use. Default is 10.')
    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(args.folder_path, args.num_threads)