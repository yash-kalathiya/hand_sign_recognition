import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import streamlit as st
from Speak import SpeakWindow  # Ensure SpeakWindow is implemented

def SignDetection(box):
    detector = HandDetector(maxHands=1)
    classifier = Classifier("hand_sign_with_digits_mobilenetv2.h5", "labels.txt")
    
    offset = 20
    imgSize = 224  # Adjusted to match the input size of the model
    labels = ["0","1","2","3","4","5","6","7","8","9","a", "b", "c", "d", "del", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", " ", "t", "u", "v", "w", "x", "y", "z"]
    
    copy_last_word = ""
    output_sentence = ""
    prev_prediction = ""
    prev_prediction_count = 0
    frame_count = 0
    del_count = 0
    ready_for_speech = False
    
    cap = cv2.VideoCapture(0)
    
    # Streamlit UI elements
    image_placeholder = st.empty()
    stop_button = st.button("Stop", key='stop_sign')

    while True:
        success, img = cap.read()
        if not success:
            break
        imgOutput = img.copy()
        
        # Detect the hand and its bounding box
        hands, img = detector.findHands(img, draw=False)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Crop the hand from the frame using the bounding box and some padding
            x_pad = max(0, x - offset)
            y_pad = max(0, y - offset)
            w_pad = min(img.shape[1], x + w + offset) - x_pad
            h_pad = min(img.shape[0], y + h + offset) - y_pad
            imgCrop = img[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                # Resize the cropped hand to the required size (224x224)
                imgResize = cv2.resize(imgCrop, (imgSize, imgSize))

                # Pass the resized hand image to the classifier
                prediction, index = classifier.getPrediction(imgResize, draw=False)

                # Handle "del" label to delete the last character
                if labels[index] == "del":
                    del_count += 1
                    if del_count >= 7:
                        if len(output_sentence) > 0:
                            output_sentence = output_sentence[:-1]
                        del_count = 0
                else:
                    del_count = 0
                    if labels[index] != prev_prediction:
                        prev_prediction_count = 0
                    else:
                        prev_prediction_count += 1

                    if prev_prediction_count >= 10:
                        output_sentence += labels[index]
                        prev_prediction_count = 0

                prev_prediction = labels[index]

                # Display the prediction on the output frame
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                cv2.rectangle(imgOutput, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), (255, 0, 0), 4)
                cv2.putText(imgOutput, output_sentence, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Display the output frame in Streamlit
        image_placeholder.image(imgOutput, channels="BGR", use_column_width=True)

        # Stop button functionality
        if stop_button:
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

        # Handle speech if ready
        if output_sentence and ready_for_speech:
            words = output_sentence.split()
            last_word = words[-1]
            copy_last_word = last_word
            SpeakWindow(last_word.strip(), box)
            ready_for_speech = False

        if output_sentence and (output_sentence[-1] == " "):
            words = output_sentence.split()
            last_word = words[-1]
            if last_word and (last_word == copy_last_word):
                ready_for_speech = False
            else:
                ready_for_speech = True
def main(box):
    st.title("Real-Time Hand Sign Detection")
    st.write("This application detects and classifies hand signs in real time.")
    
    if st.button("Start Detection"):
        SignDetection(box)

if __name__ == "__main__":
    box = st.empty()
    SpeakWindow("Started..",box)
    main(box)
