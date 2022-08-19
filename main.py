import sys
import os
import cv2
import numpy as np
import logging
import math

from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt5.QtGui import (QBrush, QConicalGradient, QLinearGradient, QPainter,
        QPainterPath, QPalette, QPen, QPixmap, QPolygon, QRadialGradient, QFont, QColor,QImage)


from mylib import *





## global vars
history_len = 10 # number of undo history
debug = False

logger = logging.getLogger(__name__)
if debug:
    logger.setLevel(logging.DEBUG)


class ImageEditorData(object):
    '''
    class to hold data for the application
    '''
    def __init__(self):
        self.original_image = None
        self.edited_image = None
        self.history_len = history_len # how many past edits to store
         # Undo feature implemented using circular array
        self.image_history = [None]*self.history_len  # circular array of past images.
        self.current_history_index = -1  # points to current image
        self.first_history_index = -1 # points to first image in the array
        self.last_history_index = -1 # poinst to last image (useful for redo)


class RenderArea(QWidget):
    '''
    this class holds the image and allows rendering of interactive rectangle on it
    '''
    # newPoint = pyqtSignal(dict)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pen = QPen(Qt.green,4, Qt.SolidLine)
        self.pixmap = QPixmap()

        self.antialiased = False
        self.transformed = False

        self.setBackgroundRole(QPalette.Base)
        self.setAutoFillBackground(True)

        self.orgx = 0
        self.orgy = 0
        self.scale = 1.0
        self.radius = 5 # circle radius of points
        self.penwidth = 4
        self.points=None
        self.pointslist= []

        self.setPixmap = self.setBackground

    def reset(self):
        self.orgx = 0
        self.orgy = 0
        self.scale = 1.0
        self.points=None
        self.pointslist= []


    def minimumSizeHint(self):
        return QSize(100, 100)

    def sizeHint(self):
        return QSize(400, 200)


    def setPoints(self, points):
        # points are in image cordinates
        self.pointslist = points
        self.points = [QPoint(*i) for i in points]
        self.update()

    def getPoints(self):
        # points = [(i[0]-self.orgx,i[1]-self.orgy) for i in self.pointslist]
        return self.pointslist

    def setBackground(self, pm):
        self.pixmap = pm
        self.update()

    def setPen(self, pen):
        self.pen = pen
        self.update()

    def setBrush(self, brush):
        self.brush = brush
        self.update()

    def setAntialiased(self, antialiased):
        self.antialiased = antialiased
        self.update()

    def setTransformed(self, transformed):
        self.transformed = transformed
        self.update()

    def paintEvent(self, event):

        W,H = self.width(),self.height()
        h,w = self.pixmap.height(), self.pixmap.width()
        if h>0 and w>0:
            self.scale = 0.8* min(W/w,H/h)
        self.orgx = (W-self.scale*w)//2
        self.orgy = (H-self.scale*h)//2

        painter = QPainter(self)
        self.pen.setWidth(self.penwidth/self.scale)
        painter.translate(self.orgx, self.orgy)
        painter.scale(self.scale, self.scale)
        painter.setPen(self.pen)
        if self.antialiased:
            painter.setRenderHint(QPainter.Antialiasing)
        # painter.save()
        painter.drawPixmap(0,0,self.pixmap)
        if self.points is not None:
            painter.drawPolygon(QPolygon(self.points))
        for pt in self.pointslist:
            painter.drawEllipse(QPoint(*pt), self.radius/self.scale, self.radius/self.scale);

    def check_point_click(self,x,y,thresh = 100):
        for index,pt in enumerate(self.pointslist):
            if (x-pt[0])**2+(y-pt[1])**2 < thresh:
                return index
        return -1

    def mousePressEvent(self, event):
        pt = event.pos()
        pt -= QPoint(self.orgx,self.orgy)
        pt /= self.scale
        thresh = 100/self.scale
        pointIndex = self.check_point_click(pt.x(),pt.y(),thresh)
        if pointIndex >= 0:
            self.clickedPointIndex = pointIndex
        else:
            self.clickedPointIndex = None

        if self.clickedPointIndex is not None:
            self.points[self.clickedPointIndex] = QPoint(pt)
            self.pointslist[self.clickedPointIndex] = [pt.x(),pt.y()]
            # self.path.moveTo(event.pos())
        self.update()
    #
    def mouseMoveEvent(self, event):
        if self.clickedPointIndex is not None:
            pt = event.pos()
            pt -= QPoint(self.orgx,self.orgy)
            pt /= self.scale
            self.points[self.clickedPointIndex] = pt
            self.pointslist[self.clickedPointIndex] = [pt.x(),pt.y()]
        self.update()




class Window(QMainWindow):

    def __init__(self, parent=None):

        super().__init__(parent) # intialise super class
        loadUi('main_window.ui',self) # load main_window ui in mainwindow
        self.setWindowTitle('Document Scanner')

        self.render_area = RenderArea() # add label widget to hold image
        # self.render_area.newPoint.connect(self.increment_history)
        self.render_area.resize(300,200)
        self.setCentralWidget(self.render_area)

        # self.render_area.newPoint.connect(lambda p: print(
        #                 'Coordinates: ( %d : %d )' % (p.x(), p.y())))


        # disable undo redo save buttons. they will be renabled on right circustances
        self.actionUndo.setEnabled(False)
        self.actionRedo.setEnabled(False)
        self.actionSave.setEnabled(False)
        self.actionReset.setEnabled(False)

        # connect different actions of buttons to corresponding action handler functions
        self.actionOpen.triggered.connect(self.openfile)
        self.actionSave.triggered.connect(self.savefile)
        self.actionUndo.triggered.connect(self.undo)
        self.actionRedo.triggered.connect(self.redo)
        self.actionReset.triggered.connect(self.reset)
        self.actionQuit.triggered.connect(self.close)

        self.actionFind_Boundary.triggered.connect(self._action_find_boundary)
        self.actionCrop_Selection.triggered.connect(self._action_crop_selection)
        self.actionGrayscale.triggered.connect(self._action_grayscale)
        self.actionBinarisation.triggered.connect(self._action_binarisation)
        self.actionHistogram_Equalisation.triggered.connect(self._action_histogram_equalisation)

        self.disable_filters() # disable filters untill image is opened


        self.ie = ImageEditorData() # new instance to hold data

        self.opts ={}# default_opts # default opts from config.py is loaded


    def alert(self,text):
        msg = QMessageBox(self) # new message box
        msg.setIcon(QMessageBox.Warning) # add warning icon
        msg.setInformativeText(text) # set warning message
        msg.setWindowTitle("Alert") # set window title
        msg.exec() # show the message box


    def show_edited_image(self):
        '''
        it converts np-type array stored to pixmap format
        and shows it in fron end. after applying any filter it must be called
        to make new image visible
        '''
        # set backend edited_image to front end after converting ot pixmap.
        self.render_area.setPixmap(self.get_pixmap(self.ie.edited_image))

    def enable_filters(self):
        '''
        Once image is opened enable all filters
        '''
        for a in [self.actionGrayscale,self.actionBinarisation,self.actionHistogram_Equalisation,self.actionFind_Boundary,self.actionCrop_Selection]:
            a.setEnabled(True)

    def disable_filters(self):
        '''
        Untill image is opened disable all filters
        '''
        for a in [self.actionGrayscale,self.actionBinarisation,self.actionHistogram_Equalisation,self.actionFind_Boundary,self.actionCrop_Selection]:
            a.setEnabled(False)

    def undo(self):
        ischange = False
        ie = self.ie # image eidtor data

        # if current index is more than first index then enable Undo
        # if last index is more than current index then enable redo
        # on undo click set image at previous index to edited_image subwindow
        # on redo click set image at next index to edited image subwindow
        if(ie.first_history_index>=0):
            if(ie.current_history_index>ie.first_history_index):
                self.ie.current_history_index -= 1
                self.ie.edited_image = self.ie.image_history[self.ie.current_history_index%self.ie.history_len]
                ischange = True

                if self.ie.current_history_index <= self.ie.first_history_index:
                    self.actionUndo.setEnabled(False)
                else:
                    self.actionUndo.setEnabled(True)
                if self.ie.current_history_index<self.ie.last_history_index:
                    self.actionRedo.setEnabled(True)
                else:
                    self.actionRed.setEnabled(False)
        if ischange:
            self.show_edited_image() # if show edited image window in fron end

    def redo(self):
        ischange = False
        ie = self.ie
        # if current index is more than first index then enable Undo
        # if last index is more than current index then enable redo
        # on undo click set image at previous index to edited_image subwindow
        # on redo click set image at next index to edited image subwindow
        if(ie.first_history_index>=0):
            if(ie.current_history_index<ie.last_history_index):
                self.ie.current_history_index += 1
                self.ie.edited_image = self.ie.image_history[self.ie.current_history_index%self.ie.history_len]
                ischange = True


            if self.ie.current_history_index>=self.ie.last_history_index:
                self.actionRedo.setEnabled(False)
            else:
                self.actionRedo.setEnabled(True)
            if self.ie.current_history_index <= self.ie.first_history_index:
                self.actionUndo.setEnabled(False)
            else:
                self.actionUndo.setEnabled(True)

        if ischange:
            self.show_edited_image() # show image in front end

    def reset(self):
        self.ie.first_history_index = -1 # reset history index
        self.ie.edited_image = np.array(self.ie.original_image) # set original image to edited image
        self.show_edited_image()
        self.increment_history() #update history arrays
        self.render_area.reset()

    def increment_history(self,event=None):
        '''
        updates history arrays and indices
        if any filter is applied, it should be called to store
        current new image in the history array
        '''
        if self.ie.first_history_index<0:
            self.ie.first_history_index = 0
            self.ie.current_history_index = 0
            self.ie.last_history_index = 0
        else:
            self.ie.current_history_index+=1
            self.ie.last_history_index = self.ie.current_history_index
        # store current new image in history array
        if event is None:
            event = np.array(self.ie.edited_image) #{'type':'img','data':np.array(self.ie.edited_image)}
        self.ie.image_history[self.ie.current_history_index%self.ie.history_len] = event #np.array(self.ie.edited_image)

        # check and enable undo
        if self.ie.current_history_index > self.ie.first_history_index:
            self.actionUndo.setEnabled(True)

    def openfile(self):

        if debug:
            fname = ('data/test.jpg','lulu')
        else:
            fname = QFileDialog.getOpenFileName(self, 'Open file',
             '.',"Image (*.png *.jpg *.jpeg *.jp2 *.bmp);;All files (*)")
        if len(fname) and len(fname[0]):
            fname = fname[0]
            # image = cv2.imread(fname,cv2.IMREAD_UNCHANGED)
            image = cv2.imread(fname)

            if image is None:
                self.alert("Uable to open file %s"%fname)
                return 1
            logger.debug('shape: %s dtype: %s'%(image.shape,image.dtype))
            # if len(image.shape) == 3 and image.shape[-1] == 4:
            #     image = cv2.cvtColor(image,cv2.CV_BGRA2BGR)
            self.ie.original_image = image
            self.ie.edited_image = np.array(image)
            self.render_area.reset()
            # pixmap = QPixmap(fname)
            pixmap = self.get_pixmap(self.ie.edited_image)
            # self.label_oimg.setPixmap(pixmap)
            self.render_area.setPixmap(pixmap)
            # self.label_oimg.resize(pixmap.width(),pixmap.height())
            # self.render_area.resize(pixmap.width(),pixmap.height())

            self.ie.first_history_index = -1
            self.increment_history()
            self.actionSave.setEnabled(True)
            self.actionReset.setEnabled(True)
            self.enable_filters()
            # self.update_opts() # change filter parameters based on image type

    def savefile(self):

        name = QFileDialog.getSaveFileName(self, 'Save Image')
        name = name[0]
        if not len(name):
            return 1
        ext = os.path.splitext(name)[-1]
        if not ext:
            self.alert('No extension provided')
            return 1
        logger.debug(name)

        try:
            cv2.imwrite(name,self.ie.edited_image)
        except Exception as e:
            self.alert("Error: Unable to save image.\n\n"+str(e))

    def get_pixmap(self,cv_image):
        '''
        converts cv_image (numpy array) to pixmap format (pyqt's image container)
        '''
        height, width, channel = cv_image.shape
        bytesPerLine = 3 * width
        qImg = QImage(cv_image.data, width, height, bytesPerLine, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qImg)
        return pixmap


    def _action_find_boundary(self):
        '''
        automatically find the boudary and show it in UI
        '''
        boundary = detect_boundary(self.ie.edited_image) # 'edge_detection', 'intensity_thresh'
        self.show_boundary(boundary)
        self.increment_history()

    def show_boundary(self,boundary):
        '''
        sets the detected points in RenderArea which renders it as polygon
        '''
        print(boundary)
        boundary = boundary.tolist()
        self.render_area.setPoints(boundary)

    def _action_crop_selection(self):
        '''
        take points given by user and crop and rectify the image to the document
        '''
        new_image = crop_and_transform(self.ie.edited_image,self.render_area.getPoints())
        self.ie.edited_image = new_image

        # self.render_area.setPixmap(self.get_pixmap(new_image) )
        self.render_area.pointslist = []
        self.render_area.points = None
        self.show_edited_image()
        self.increment_history()

    def _action_grayscale(self):
        '''
        convert to grayscale
        '''
        self.ie.edited_image =  cv.cvtColor(self.ie.edited_image,cv.COLOR_BGR2GRAY)
        self.ie.edited_image = cv.cvtColor(self.ie.edited_image,cv.COLOR_GRAY2BGR)
        # self.render_area.setPixmap(self.get_pixmap(self.ie.edited_image) )
        self.show_edited_image()
        self.increment_history()

    # uses gaussian thresholding for binarisation
    def _action_binarisation(self):
        '''
        binarise using adaptive threshold (gaussian)
        '''
        self.ie.edited_image =  cv.cvtColor(self.ie.edited_image,cv.COLOR_BGR2GRAY)
        # val,self.ie.edited_image =  cv.threshold(self.ie.edited_image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
        self.ie.edited_image = cv.adaptiveThreshold(self.ie.edited_image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
        self.ie.edited_image = cv.cvtColor(self.ie.edited_image,cv.COLOR_GRAY2BGR)
        # self.render_area.setPixmap(self.get_pixmap(self.ie.edited_image) )
        self.show_edited_image()
        self.increment_history()

    def _action_histogram_equalisation(self):

        image = self.ie.edited_image
        image_he = equalizeHist_transform(image)
        self.ie.edited_image = image_he # np.stack([image_he,image_he,image_he],axis=-1)
        # print(self.ie.edited_image.shape)
        self.show_edited_image()
        self.increment_history()


###                                  ###
###  Functions to perform filtering  ###
###                                  ###

def equalizeHist_transform(image,use_cv=True):
    if len(image.shape) == 2:
        h,w = image.shape
        c=1
    elif len(image.shape) == 3:
        h,w,c = image.shape
    if c==1:
        if use_cv:
            return cv2.equalizeHist(image)
        else:
            return equalizeHist(image) # histogram equalization
    elif c==3:
        # convert bgr image to hsv and apply transform only on v-channel
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if use_cv:
            hsv_image[:,:,-1]=cv2.equalizeHist(hsv_image[:,:,-1])
        else:
            hsv_image[:,:,-1]=equalizeHist(hsv_image[:,:,-1])

        # convert hsv image back to rgb
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    else:
        raise ValueError('Invalid channel number: %d'%c)

if __name__ == "__main__":

    app = QApplication(sys.argv) # start aan instance of application

    win = Window() # create new window

    win.show() # show window
    ret = app.exec() #run the application
    sys.exit(ret) # exit
