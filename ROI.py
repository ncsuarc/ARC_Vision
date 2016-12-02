class ROI():
    def __init__(self, arc_image, image, x=0, y=0, width=0, height=0):
        self.arc_image = arc_image
        self.image = image
        self.roi = image[y:y+height, x:x+width]
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.roi = image[y:y+height, x:x+width]

