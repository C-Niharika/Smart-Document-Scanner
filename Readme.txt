### Document Scanner

This is a tool which can automatically extract document from an image containing both document and the backgroud. For this it uses canny edge detection followed by contour detection. From contours we select the largest one and find convex hull around. Then we approximate this convex hull with lower order polynomial (a quadrilateral). This is the boundary of the document. From this we compute the perspective transform matrix and extract and correct the distortion in the image.

## Installation

python : 3.6.9
os: Ubuntu 18.04

For GUI pyqt5 library along with qt designer is used

To install python dpendencies run
    pip install -r requirements.txt
    
To start the app go to project dir in terminal and run
    python main.py
    

To compile .ui files to .py files use
for i in *.ui; do echo $i;pyuic5 $i -o ${i%.ui}.py;done



