#Dependencies
import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Constants
WIDTH = 640
HEIGHT = 480
BUBBLE_RADIUS = 30
BUBBLE_SPEED = 10
HAND_COLOR = (0, 110, 255)
PARTICLE_COUNT = 20
PARTICLE_SPEED = 25
PARTICLE_RADIUS = 2
COUNT = 0
time_limit = 15


# Create bubble class
class Bubble:
    def __init__(self):
        self.radius = random.randint(BUBBLE_RADIUS-12,BUBBLE_RADIUS+12)
        self.center = (random.randint(self.radius, WIDTH - self.radius),
                       random.randint(self.radius, HEIGHT - self.radius))
        BUBBLE_COLOR = (random.randint(220, 255), random.randint(186, 226), random.randint(125, 145))
        self.color = BUBBLE_COLOR
        self.speed = BUBBLE_SPEED
        self.direction = (random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5))
        self.direction = np.array(self.direction) / np.linalg.norm(self.direction)

    def update(self):
        # Move bubble
        self.center += self.direction * (self.speed)/2

        # Bounce off walls
        if self.center[0] < self.radius or self.center[0] > WIDTH - self.radius:
            self.direction[0] *= -1
        if self.center[1] < self.radius or self.center[1] > HEIGHT - self.radius:
            self.direction[1] *= -1

    def draw(self, image):
        # Draw bubble
        cv2.circle(image, (int(self.center[0]), int(self.center[1])),
                   self.radius, self.color, -1)
        # # overlay = image.copy()
        # alpha = 0.7
        # cv2.addWeighted(image, alpha, image, 1 - alpha, 0)



# Create particle class
class Particle:
    def __init__(self, center, direction):
        self.center = center
        self.color = Bubble().color
        self.speed = PARTICLE_SPEED
        self.direction = direction
        self.direction = np.array(self.direction) / np.linalg.norm(self.direction)

    def update(self):
        # Move particle
        self.center += self.direction * self.speed

    def draw(self, image):
        # Draw particle
        cv2.circle(image, (int(self.center[0]), int(self.center[1])),
                   PARTICLE_RADIUS, self.color, -1)


# Create Mediapipe hands object
mp_hands = mp.solutions.hands.Hands()

# Create bubbles list
bubbles = []

# Create particles list
particles = []

# Create video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Run simulation
start = time.time()
while True:
    # Capture frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    time_left = time_limit - (time.time() - start)

    if time_left > 0:
        cv2.putText(frame, "Time left {:.2f}".format(time_left), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    cv2.LINE_4)

        # Detect hands in frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image)

        # Update bubbles
        for bubble in bubbles:
            bubble.update()
            bubble.draw(frame)

        # Generate new bubbles
        if random.random() < 0.08:
            bubbles.append(Bubble())

        # Check if hand is touching bubbles
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for bubble in bubbles:
                    dist = np.linalg.norm(np.array([hand_landmarks.landmark[8].x * WIDTH,
                                                    hand_landmarks.landmark[8].y * HEIGHT]) - bubble.center)
                    if dist < bubble.radius:
                        # Bubble burst effect
                        COUNT += 1
                        bubbles.remove(bubble)
                        for i in range(PARTICLE_COUNT):
                            direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
                            particles.append(Particle(bubble.center, direction))

                            # Update particles
                            for particle in particles:
                                particle.update()
                                particle.draw(frame)

                                # Remove particle if it goes out of screen
                                if particle.center[0] < 0 or particle.center[0] > WIDTH or particle.center[1] < 0 or particle.center[1] > HEIGHT:
                                    particles.remove(particle)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks,
                                                          mp.solutions.hands.HAND_CONNECTIONS,
                                                          landmark_drawing_spec=mp.solutions.drawing_utils
                                                          .DrawingSpec(color=HAND_COLOR))

        cv2.putText(frame, "Score: " + str(COUNT), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), cv2.LINE_4)

    else:
        cv2.putText(frame, "Game Over!!", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), cv2.LINE_4)
        cv2.putText(frame, "Your Score: {}".format(COUNT), (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), cv2.LINE_4)
    # Display frame
    cv2.imshow('Bubble Pop', frame)

    # Exit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) == ord('r'):
        start = time.time()
        COUNT = 0

# Release video capture
cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()