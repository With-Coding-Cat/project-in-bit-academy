#파이토치 사용시 CUDA error: initialization error 가 나타날 수 있음. 이때 하단의 코드를 파이썬 파일 제일 앞에 삽입하기(GPU 할당 관련 문제)
torch.multiprocessing.set_start_method('spawn')

#텐서플로에서 GPU로 실행시 CUDA 관련 오류시 하단의 코드를 파이썬 제일 앞에 삽입하기(마찬가지로 GPU 할당 관련 문제
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
