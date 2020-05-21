# gesture-recognition
Gesture Recognition by Fingertip Calculation  and Hand Tracking

We used OpenCV and C++ to create a program which takes camera view as input and segments the user’s hand from this image to 
calculate number gestures. Initially the background and the user’s hand color is sampled. The hand segmentation’s result is 
improved by using canny edge detection and is then used to create a bounding box. This bounding box is subtracted from the 
result of the background segmentation to eliminate non-hand objects. The user’s number gestures up to five can be recognized 
by mainly calculating and counting the cavities between fingers, and the location of the hand is tracked. We used low resolutions
in hand recognition to increase the execution speed of the hand detection system, it performs with above 15 frames per second which 
can be considered real-time. 

Overall if the segmentation process is successful the program works as desired, but the user’s surrounding can greatly reduce the 
accuracy of segmentation. The result can be improved by using Leap or Kinect instead of a webcam. We did not want to develop our 
program using these tools because we wanted to write our own segmentation algorithm and webcams are abundant when compared to these 
tools. 
