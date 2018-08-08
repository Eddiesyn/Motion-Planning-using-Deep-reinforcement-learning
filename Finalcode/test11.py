
def sigint_handler(signum, frame):
  print('system terminate')


signal.signal(signal.SIGINT, sigint_handler)

