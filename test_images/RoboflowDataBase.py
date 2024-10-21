from roboflow import Roboflow
format = "yolov7"
rf = Roboflow(api_key="ITupacAwsoZCNY7yGjes") #teoretycznie jest to prywatne ale z racji tego że wszystkie projekty są i tak publiczne to mam wywalone
project = rf.workspace("firsttry-ekt4d").project("crosswalksai") #database name
version = project.version(1) #wersion
dataset = version.download(format, location="./movmentAidImgs") #format and location

project = rf.workspace("firsttry-ekt4d").project("crosswalkai-zebracrossing")
version = project.version(1)
dataset = version.download(format, location="./zebraCrossing")

project = rf.workspace("firsttry-ekt4d").project("eazy-face-detection")
version = project.version(1)
dataset = version.download(format, location="./EazyFace")

project = rf.workspace("firsttry-ekt4d").project("medium-face-deteciton")
version = project.version(1)
dataset = version.download(format, location="./MidFace")

project = rf.workspace("firsttry-ekt4d").project("hard-face-reco")
version = project.version(1)
dataset = version.download(format, location="./HardFace")