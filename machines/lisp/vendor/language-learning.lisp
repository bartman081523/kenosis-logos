;;;; language-learning.lisp
;;;; Learn user's language patterns from interactions

(in-package #:lisp-cognito-philosopher)

;;; ============================================================================
;;; LANGUAGE LEARNING
;;; ============================================================================

(defun learn-from-input (cognito user-input)
  "Learn from user input by updating state based on input patterns.
   This creates a connection between user's words and internal state."
  
  ;; Extract words from input
  (let* ((words (extract-words user-input))
         (word-vectors (mapcar #'word-to-vector words))
         (avg-vector (average-vectors word-vectors)))
    
    ;; Update state by blending with learned pattern
    (when avg-vector
      (let ((state (cognito-construct-state cognito)))
        (dotimes (i (min (length state) (length avg-vector)))
          (setf (aref state i)
                (+ (* 0.8 (aref state i))
                   (* 0.2 (aref avg-vector i))))))
      
      (format t "📚 Learned from: ~A~%" user-input))))

(defun extract-words (text)
  "Extract words from text (lowercase, remove punctuation)."
  (let ((cleaned (string-downcase 
                  (remove-if-not #'(lambda (c) 
                                    (or (alpha-char-p c) 
                                        (char= c #\Space)))
                                text))))
    (remove-if #'(lambda (s) (string= s ""))
               (split-string cleaned #\Space))))

(defun word-to-vector (word)
  "Convert a word to a vector deterministically.
   Each character contributes to the vector."
  (let ((vec (make-array 8 :initial-element 0.0d0)))
    (loop for char across word
          for i from 0
          do (let ((char-val (/ (char-code char) 127.0)))
               (incf (aref vec (mod i 8)) char-val)))
    
    ;; Normalize
    (let ((magnitude (vector-norm vec)))
      (when (> magnitude 0)
        (dotimes (i (length vec))
          (setf (aref vec i) (/ (aref vec i) magnitude)))))
    
    vec))

(defun average-vectors (vectors)
  "Average multiple vectors."
  (when vectors
    (let ((result (make-array 8 :initial-element 0.0d0))
          (count (length vectors)))
      (dolist (vec vectors)
        (dotimes (i (min 8 (length vec)))
          (incf (aref result i) (aref vec i))))
      (dotimes (i 8)
        (setf (aref result i) (/ (aref result i) count)))
      result)))

;;; ============================================================================
;;; RESPONSE GENERATION
;;; ============================================================================

(defun determine-response-length (cognito)
  "Determine how many words to generate based on state magnitude.
   Higher magnitude = longer responses."
  (let* ((state (cognito-construct-state cognito))
         (magnitude (vector-norm state)))
    (cond
      ((< magnitude 0.5) 1)   ; Very short
      ((< magnitude 1.0) 2)   ; Short
      ((< magnitude 1.5) 3)   ; Medium
      ((< magnitude 2.0) 4)   ; Long
      (t 5))))                ; Very long

(defun generate-response (cognito user-input)
  "Generate a response based on current state and user input.
   The response reflects what the philosopher has learned.
   Length is determined by internal state."
  
  ;; Think about the input
  (think cognito user-input)
  
  ;; Reflect
  (reflect cognito)
  
  ;; Determine response length from state
  (let* ((num-words (determine-response-length cognito))
         (gap (get-goedel-gap cognito))
         (state (cognito-construct-state cognito))
         (words nil))
    
    ;; Generate multiple words
    (dotimes (i num-words)
      (let* ((offset (* i 2))
             (sub-state (if (< (+ offset 3) (length state))
                           (subseq state offset (min (+ offset 3) (length state)))
                           state)))
        (push (vector-to-syllables sub-state) words)))
    
    ;; Combine into sentence
    (format nil "~{~A~^ ~} (gap: ~,3F)" (nreverse words) gap)))

(defun split-string (string delimiter)
  "Split string by delimiter."
  (loop with result = nil
        with start = 0
        for pos = (position delimiter string :start start)
        do (push (subseq string start pos) result)
           (if pos
               (setf start (1+ pos))
               (return (nreverse result)))
        finally (return (nreverse result))))
