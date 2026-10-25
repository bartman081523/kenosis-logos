;;;; goedel-encoder.lisp
;;;; Gödel encoding and gap computation for self-reference

(in-package #:lisp-cognito-philosopher)

;;; ============================================================================
;;; GÖDEL ENCODER
;;; ============================================================================

(defparameter *primes* '(2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71)
  "Prime numbers for Gödel encoding")

(defun goedel-encode (s-expr)
  "Encode an S-expression as a Gödel number using prime factorization.
   Truncates to avoid astronomical numbers."
  (let ((str (prin1-to-string s-expr)))
    (goedel-encode-string str)))

(defun goedel-encode-string (string &optional (max-chars 15))
  "Encode a string as a Gödel number."
  (let ((result 1)
        (truncated (subseq string 0 (min (length string) max-chars))))
    (loop for char across truncated
          for i from 0
          for prime = (nth (mod i (length *primes*)) *primes*)
          do (setf result (* result (expt prime (char-code char)))))
    (mod result (expt 10 12)))) ; Keep manageable

(defun compute-structural-signature (s-expr)
  "Extract structural features of an S-expression.
   Returns a hash table with features."
  (let ((features (make-hash-table :test 'equal)))
    (setf (gethash "depth" features) (compute-depth s-expr))
    (setf (gethash "size" features) (compute-size s-expr))
    (setf (gethash "atoms" features) (count-atoms s-expr))
    (setf (gethash "lists" features) (count-lists s-expr))
    (setf (gethash "symbols" features) (count-symbols s-expr))
    features))

(defun compute-depth (s-expr)
  "Compute maximum nesting depth of S-expression."
  (cond
    ((atom s-expr) 0)
    (t (1+ (reduce #'max (mapcar #'compute-depth s-expr) :initial-value 0)))))

(defun compute-size (s-expr)
  "Compute total number of nodes in S-expression."
  (cond
    ((atom s-expr) 1)
    (t (1+ (reduce #'+ (mapcar #'compute-size s-expr) :initial-value 0)))))

(defun count-atoms (s-expr)
  "Count number of atoms in S-expression."
  (cond
    ((atom s-expr) 1)
    (t (reduce #'+ (mapcar #'count-atoms s-expr) :initial-value 0))))

(defun count-lists (s-expr)
  "Count number of lists in S-expression."
  (cond
    ((atom s-expr) 0)
    (t (1+ (reduce #'+ (mapcar #'count-lists s-expr) :initial-value 0)))))

(defun count-symbols (s-expr)
  "Count number of symbols in S-expression."
  (cond
    ((symbolp s-expr) 1)
    ((atom s-expr) 0)
    (t (reduce #'+ (mapcar #'count-symbols s-expr) :initial-value 0))))

(defun predict-output-signature (s-expr)
  "Attempt to predict what the S-expression will output.
   This is the 'computable' part. The gap between this prediction
   and actual output is the Gödel Gap.
   
   Returns a simple hash based on structure."
  (let ((sig (compute-structural-signature s-expr)))
    (sxhash (format nil "~A-~A-~A" 
                    (gethash "depth" sig)
                    (gethash "size" sig)
                    (gethash "symbols" sig)))))

;;; ============================================================================
;;; GÖDEL GAP COMPUTATION
;;; ============================================================================

(defun compute-goedel-gap (prediction actual)
  "Compute the Gödel Gap between prediction and actual state.
   
   Gap = |prediction - actual| / |actual|
   
   Low Gap = Good self-knowledge
   High Gap = Blind spot (potential for discovery)
   
   Inputs can be vectors or scalars."
  (cond
    ;; Both vectors
    ((and (vectorp prediction) (vectorp actual))
     (compute-vector-gap prediction actual))
    ;; Both numbers
    ((and (numberp prediction) (numberp actual))
     (compute-scalar-gap prediction actual))
    ;; Mixed or other types - convert to strings and compare
    (t
     (compute-string-gap (prin1-to-string prediction)
                        (prin1-to-string actual)))))

(defun compute-vector-gap (pred-vec actual-vec)
  "Compute gap between two vectors."
  (let ((diff-norm (vector-distance pred-vec actual-vec))
        (actual-norm (vector-norm actual-vec)))
    (if (< actual-norm 1.0d-9)
        0.0d0
        (/ diff-norm actual-norm))))

(defun compute-scalar-gap (pred actual)
  "Compute gap between two scalars."
  (let ((diff (abs (- pred actual)))
        (magnitude (abs actual)))
    (if (< magnitude 1.0d-9)
        0.0d0
        (/ diff magnitude))))

(defun compute-string-gap (pred-str actual-str)
  "Compute gap between two strings using edit distance."
  (let ((edit-dist (levenshtein-distance pred-str actual-str))
        (max-len (max (length pred-str) (length actual-str))))
    (if (zerop max-len)
        0.0d0
        (/ edit-dist max-len))))

(defun levenshtein-distance (str1 str2)
  "Compute Levenshtein edit distance between two strings."
  (let* ((len1 (length str1))
         (len2 (length str2))
         (d (make-array (list (1+ len1) (1+ len2)) :initial-element 0)))
    ;; Initialize first row and column
    (dotimes (i (1+ len1))
      (setf (aref d i 0) i))
    (dotimes (j (1+ len2))
      (setf (aref d 0 j) j))
    ;; Compute distances
    (dotimes (i len1)
      (dotimes (j len2)
        (let ((cost (if (char= (char str1 i) (char str2 j)) 0 1)))
          (setf (aref d (1+ i) (1+ j))
                (min (1+ (aref d i (1+ j)))           ; deletion
                     (1+ (aref d (1+ i) j))           ; insertion
                     (+ (aref d i j) cost))))))       ; substitution
    (aref d len1 len2)))

;;; ============================================================================
;;; GAP INTERPRETATION
;;; ============================================================================

(defun interpret-gap (gap)
  "Interpret the meaning of a Gödel Gap value.
   Returns a keyword indicating the level of self-knowledge."
  (cond
    ((< gap 0.1) :excellent-self-knowledge)
    ((< gap 0.3) :good-self-knowledge)
    ((< gap 0.5) :moderate-blind-spot)
    ((< gap 0.7) :significant-blind-spot)
    (t :severe-blind-spot)))

(defun gap-description (gap)
  "Get a human-readable description of a Gödel Gap value."
  (let ((interpretation (interpret-gap gap)))
    (case interpretation
      (:excellent-self-knowledge "System has excellent self-model")
      (:good-self-knowledge "System has good self-knowledge")
      (:moderate-blind-spot "System has moderate blind spots")
      (:significant-blind-spot "System has significant blind spots - potential for discovery")
      (:severe-blind-spot "System behavior is highly unpredictable to itself")
      (t "Unknown gap level"))))
