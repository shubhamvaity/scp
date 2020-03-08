import cv2 
import pytesseract
import os

for filename in os.listdir('.\op2'):
	if filename.endswith(".jpg") or filename.endswith(".png"):
		print(filename)
		image = cv2.imread("C:\\Users\\Shubham Vaity\\Desktop\\Hackathon\\TEST 2\\op2\\"+filename)
		# Adding custom options
		custom_config = r'--oem 3 --psm 6'
		try:
			text = pytesseract.image_to_string(image, config=custom_config)
			os.chdir("C:\\Users\\Shubham Vaity\\Desktop\\Hackathon\\TEST 2\\ocrop")
			f1 = open(filename[0:8]+".txt","w+")
			f1.write(text)
			f1.close()
		except:
			print('blank page')
