import os
for filename in os.listdir():
	size = os.path.splitext(filename)[0]
	print(filename)
	if(size != "sidewindow" and size != "ori"):
		os.rename(filename, size.zfill(4))