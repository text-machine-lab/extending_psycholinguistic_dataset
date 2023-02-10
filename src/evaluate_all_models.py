import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("datasetpath", default=None, type=str)
parser.add_argument("--prediction", default=False, type=str)


args = parser.parse_args()

os.system('python evaluation.py {} bert-base-uncased --prediction {}'.format(args.datasetpath, args.prediction))
os.system('python evaluation.py {} bert-large-uncased --prediction {}'.format(args.datasetpath, args.prediction))

os.system('python evaluation.py {} distilbert-base-uncased --prediction {}'.format(args.datasetpath, args.prediction))

os.system('python evaluation.py {} roberta-base --prediction {}'.format(args.datasetpath, args.prediction))
os.system('python evaluation.py {} roberta-large --prediction {}'.format(args.datasetpath, args.prediction))

os.system('python evaluation.py {} albert-base-v1 --prediction {}'.format(args.datasetpath, args.prediction))
os.system('python evaluation.py {} albert-large-v1 --prediction {}'.format(args.datasetpath, args.prediction))
os.system('python evaluation.py {} albert-xlarge-v1 --prediction {}'.format(args.datasetpath, args.prediction))
os.system('python evaluation.py {} albert-xxlarge-v1 --prediction {}'.format(args.datasetpath, args.prediction))

os.system('python evaluation.py {} albert-base-v2 --prediction {}'.format(args.datasetpath, args.prediction))
os.system('python evaluation.py {} albert-large-v2 --prediction {}'.format(args.datasetpath, args.prediction))
os.system('python evaluation.py {} albert-xlarge-v2 --prediction {}'.format(args.datasetpath, args.prediction))
os.system('python evaluation.py {} albert-xxlarge-v2 --prediction {}'.format(args.datasetpath, args.prediction))

os.system('python evaluation.py {} t5-small --prediction {}'.format(args.datasetpath, args.prediction))
os.system('python evaluation.py {} t5-base --prediction {}'.format(args.datasetpath, args.prediction))
os.system('python evaluation.py {} t5-large --prediction {}'.format(args.datasetpath, args.prediction))
os.system('python evaluation.py {} t5-3b --prediction {}'.format(args.datasetpath, args.prediction))

os.system('python evaluation.py {} gpt2 --prediction {}'.format(args.datasetpath, args.prediction))
os.system('python evaluation.py {} gpt2-medium --prediction {}'.format(args.datasetpath, args.prediction))
os.system('python evaluation.py {} gpt2-large --prediction {}'.format(args.datasetpath, args.prediction))
# os.system('python evaluation.py {} EleutherAI/gpt-neo-1.3B'.format(args.datasetpath, args.prediction))

os.system('python evaluation.py {} gpt2-xl --prediction {}'.format(args.datasetpath, args.prediction))