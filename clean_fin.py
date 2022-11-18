import sys
import unicodedata
import six

for line in sys.stdin:
    line = line.rstrip().replace('#', '')
    print(u'%s' % line)
