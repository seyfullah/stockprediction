import time
import datetime

start = time.time()
print("start:" + str(datetime.datetime.now()))
end = time.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("end:  " + str(datetime.datetime.now()))
print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
