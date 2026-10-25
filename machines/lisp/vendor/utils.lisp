;;;; utils.lisp
;;;; Utility functions for Lisp Cognito Philosopher

(in-package #:lisp-cognito-philosopher)

;;; ============================================================================
;;; JSON UTILITIES (Stub implementations - would use cl-json in production)
;;; ============================================================================

(defun load-json-file (filepath)
  "Load and parse a JSON file. Returns parsed data or NIL on error.
   NOTE: This is a stub implementation. In production, use cl-json."
  (handler-case
      (with-open-file (stream filepath :direction :input :if-does-not-exist nil)
        (when stream
          ;; Stub: just return empty alist
          (log-message :warn "JSON loading is stubbed - would need cl-json for real loading")
          nil))
    (error (e)
      (format t "~&⚠️  Error loading JSON file ~A: ~A~%" filepath e)
      nil)))

(defun save-json-file (filepath data)
  "Save data to a JSON file.
   NOTE: This is a stub implementation. In production, use cl-json."
  (handler-case
      (with-open-file (stream filepath :direction :output
                              :if-exists :supersede
                              :if-does-not-exist :create)
        (format stream "~S" data)
        t)
    (error (e)
      (format t "~&❌ Error saving JSON file ~A: ~A~%" filepath e)
      nil)))

;;; ============================================================================
;;; VECTOR OPERATIONS
;;; ============================================================================

;;; ============================================================================
;;; DETERMINISTIC RANDOM NUMBER GENERATOR
;;; ============================================================================

(defparameter *rng-state* 123456789
  "State for the deterministic PRNG.")

(defun set-rng-seed (seed)
  "Set the seed for the PRNG."
  (setf *rng-state* (logand seed #xFFFFFFFF)))

(defun deterministic-random (limit)
  "Generate a deterministic random number up to limit (exclusive).
   Uses a Linear Congruential Generator (LCG)."
  (let ((a 1664525)
        (c 1013904223)
        (m 4294967296)) ; 2^32
    (setf *rng-state* (mod (+ (* a *rng-state*) c) m))
    (if (floatp limit)
        (* limit (/ (float *rng-state* 1.0d0) m))
        (mod *rng-state* limit))))

;;; ============================================================================
;;; VECTOR OPERATIONS
;;; ============================================================================

(defun make-random-vector (size &optional (mean 0.0) (stddev 1.0))
  "Create a random vector with normal distribution."
  (let ((vec (make-array size :element-type 'double-float)))
    (dotimes (i size vec)
      (setf (aref vec i) (+ mean (* stddev (random-normal)))))))

(defun random-normal ()
  "Generate a random number from standard normal distribution (Box-Muller).
   REPLACED system RANDOM with DETERMINISTIC-RANDOM."
  (let ((u1 (deterministic-random 1.0d0))
        (u2 (deterministic-random 1.0d0)))
    ;; Handle case where u1 is 0 to avoid log domain error
    (if (< u1 1.0d-10) (setf u1 1.0d-10))
    (* (sqrt (* -2.0d0 (log u1)))
       (cos (* 2.0d0 pi u2)))))

(defun vector-norm (vec)
  "Compute L2 norm of a vector."
  (sqrt (loop for x across vec sum (* x x))))

(defun normalize-vector (vec)
  "Normalize a vector to unit length."
  (let ((norm (vector-norm vec)))
    (if (< norm 1.0d-9)
        vec
        (map 'vector (lambda (x) (/ x norm)) vec))))

(defun dot-product (vec1 vec2)
  "Compute dot product of two vectors."
  (loop for x across vec1
        for y across vec2
        sum (* x y)))

(defun vector-distance (vec1 vec2)
  "Compute Euclidean distance between two vectors."
  (sqrt (loop for x across vec1
              for y across vec2
              sum (expt (- x y) 2))))

;;; ============================================================================
;;; HASH TABLE UTILITIES
;;; ============================================================================

(defun make-knowledge-base ()
  "Create a new knowledge base (hash table with string keys)."
  (make-hash-table :test 'equal))

(defun kb-get (kb key &optional default)
  "Get value from knowledge base."
  (gethash key kb default))

(defun kb-put (kb key value)
  "Put value into knowledge base."
  (setf (gethash key kb) value))

(defun kb-keys (kb)
  "Get all keys from knowledge base."
  (loop for key being the hash-keys of kb collect key))

;;; ============================================================================
;;; LOGGING UTILITIES
;;; ============================================================================

(defparameter *log-level* :info
  "Current logging level. One of :debug :info :warn :error")

(defun log-message (level format-string &rest args)
  "Log a message at the specified level."
  (when (should-log-p level)
    (let ((prefix (case level
                    (:debug "🔍")
                    (:info "ℹ️ ")
                    (:warn "⚠️ ")
                    (:error "❌")
                    (t "  "))))
      (format t "~&~A ~?" prefix format-string args))))

(defun should-log-p (level)
  "Check if a message at this level should be logged."
  (let ((levels '(:debug :info :warn :error))
        (current-idx (position *log-level* '(:debug :info :warn :error)))
        (msg-idx (position level '(:debug :info :warn :error))))
    (and current-idx msg-idx (>= msg-idx current-idx))))

(defmacro with-log-level (level &body body)
  "Execute body with a specific log level."
  `(let ((*log-level* ,level))
     ,@body))

;;; ============================================================================
;;; MATH UTILITIES
;;; ============================================================================

(defun safe-log (x)
  "Compute log with safety check for non-positive values."
  (if (<= x 0)
      0.0d0
      (log x)))

(defun safe-sqrt (x)
  "Compute sqrt with safety check for negative values."
  (if (< x 0)
      0.0d0
      (sqrt x)))

(defun safe-divide (numerator denominator &optional (default 0.0d0))
  "Divide with safety check for zero denominator."
  (if (< (abs denominator) 1.0d-9)
      default
      (/ numerator denominator)))

(defun clamp (value min-val max-val)
  "Clamp value between min and max."
  (max min-val (min max-val value)))

;;; ============================================================================
;;; STRING UTILITIES
;;; ============================================================================

(defun string-gematria (str)
  "Compute gematria (sum of character codes) of a string."
  (loop for char across str sum (char-code char)))

(defun truncate-string (str max-length &optional (suffix "..."))
  "Truncate string to max-length, adding suffix if truncated."
  (if (<= (length str) max-length)
      str
      (concatenate 'string
                   (subseq str 0 (- max-length (length suffix)))
                   suffix)))

;;; ============================================================================
;;; FILE UTILITIES
;;; ============================================================================

(defun file-exists-p (filepath)
  "Check if a file exists."
  (probe-file filepath))

(defun ensure-directory (dirpath)
  "Ensure a directory exists, creating it if necessary."
  (ensure-directories-exist dirpath))
