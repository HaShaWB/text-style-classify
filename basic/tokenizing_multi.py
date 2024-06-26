import pandas as pd
import os, gc
from concurrent.futures import ProcessPoolExecutor
import tiktoken
import multiprocessing




FILE_NUM = 20

SPLITS = 32

BATCH_SIZE = 32




# 사용할 인코딩 선택
encoding_name = "cl100k_base"  # "p50k_base", "r50k_base", "gpt2" 등으로 변경 가능

# tiktoken 토크나이저 초기화
tokenizer = tiktoken.get_encoding(encoding_name)

def tiktoken_tokenizer(sentence):
    # 문자열이 아닌 경우 빈 시리즈 반환
    if not isinstance(sentence, str):
        return pd.Series([[], []])
    
    # tiktoken을 사용하여 문장을 토큰화하고 ID로 변환
    token_ids = tokenizer.encode(sentence)
    return token_ids

def split_dataframe(df, n):
    split_dfs = []
    chunk_size = len(df) // n
    for i in range(n):
        start_index = i * chunk_size
        if i == n - 1:  # 마지막 부분은 나머지를 포함
            end_index = len(df)
        else:
            end_index = (i + 1) * chunk_size
        split_dfs.append(df.iloc[start_index:end_index].copy())
    return split_dfs

def process_dataframe(args):
    i, file, base_dir, total_steps, progress_queue = args
    df = pd.read_pickle(file)
    df["tiktoken_encodes"] = df["sentence"].apply(tiktoken_tokenizer)
    # 처리된 결과를 파일로 저장
    temp_file = os.path.join(base_dir, f"temps/temp_written_{i}.pkl")
    df.to_pickle(temp_file)
    
    progress_queue.put(1)
    
    return temp_file

def print_progress(current, total):
    progress = (current / total) * 100
    print(f"Progress: {progress:.2f}%", end='\r')

def update_progress(progress_queue, current_step, total_steps):
    while not progress_queue.empty():
        with current_step.get_lock():
            current_step.value += progress_queue.get()
            print_progress(current_step.value, total_steps)

if __name__ == "__main__":
    base_dir = os.getcwd()

    print("hello world")
    
    """    dfs = split_dataframe(pd.read_csv(os.path.join(base_dir, f"datasets/written.csv")), FILE_NUM)

        for i in range(len(dfs)):
            dfs[i].to_csv(os.path.join(base_dir, f"datasets/written_{i}.csv"), index=False)
            dfs[i] = None
            gc.collect()
        
        dfs = None
        gc.collect()"""

    print("Processing started!")

    for file_index in range(FILE_NUM):
        gc.collect()
        
        print(f"Starting the process: written_{file_index}")


        df2 = pd.read_csv(os.path.join(base_dir, f"datasets/written_{file_index}.csv"))

        


        # 데이터프레임을 128등분하여 파일로 저장
        split_df2 = split_dataframe(df2, SPLITS)

        df2 = None
        gc.collect()

        total_steps = SPLITS  # 총 스텝 수는 총 파일 수 + 총 배치 수
        current_step = multiprocessing.Value('i', 0)  # 공유 변수로 사용
        lock = multiprocessing.Lock()

        for i, df in enumerate(split_df2):
            temp_file = os.path.join(base_dir, f"temps/temp_split_{file_index}_{i}.pkl")
            df.to_pickle(temp_file)

        split_df2 = None
        gc.collect()

        num_batches = SPLITS // BATCH_SIZE
        final_results = []

        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()

        for batch_num in range(num_batches):
            batch_files = [os.path.join(base_dir, f"temps/temp_split_{file_index}_{i}.pkl") for i in range(batch_num * BATCH_SIZE, (batch_num + 1) * BATCH_SIZE)]
            
            # 배치를 멀티프로세서로 처리
            with ProcessPoolExecutor(max_workers=BATCH_SIZE) as executor:
                batch_results = list(executor.map(process_dataframe, [(i, file, base_dir, total_steps, progress_queue) for i, file in enumerate(batch_files)]))
            
            update_progress(progress_queue, current_step, total_steps)
            
            # 병렬 처리된 결과들을 결합하고 저장
            batch_df = pd.concat([pd.read_pickle(file) for file in batch_results])
            batch_output_file = os.path.join(base_dir, f"temps/batch_result_{file_index}_{batch_num}.pkl")
            batch_df.to_pickle(batch_output_file)
            
            # 임시 파일 삭제 및 메모리 해제
            for file in batch_files:
                os.remove(file)
            batch_results = None
            batch_df = None
            gc.collect()

            update_progress(progress_queue, current_step, total_steps)

            # 최종 결과 리스트에 추가
            final_results.append(batch_output_file)

        # 최종 데이터프레임 결합
        final_df_list = [pd.read_pickle(file) for file in final_results]
        final_df = pd.concat(final_df_list)
        final_df.to_pickle(os.path.join(base_dir, f"datasets/written_{file_index}.pkl"))

        # 중간 파일 삭제 및 메모리 해제
        for file in final_results:
            os.remove(file)
        final_df_list = None
        final_results = None
        final_df = None
        gc.collect()

        print_progress(total_steps, total_steps)
        print("\nProcessing complete!")

