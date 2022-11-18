import sys
import unicodedata
import six


for line in sys.stdin:
    line = line.rstrip().split()
    line[0] = line[0].split('_')[0]
    print(u'%s' % ' '.join(line))
