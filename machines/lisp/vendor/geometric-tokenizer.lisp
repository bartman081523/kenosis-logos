;;;; geometric-tokenizer.lisp
;;;; Maps words to geometric formulas using SH-72 basis

(in-package #:lisp-cognito-philosopher)

;;; ============================================================================
;;; SH-72 BASIS (The Creative Alphabet)
;;; ============================================================================

(defun get-sh-basis-formula (index-0-to-71)
  "Map index (0-71) to one of 72 fundamental basis functions.
   These represent 'topological moves' or fundamental operations."
  (let ((k (1+ (mod index-0-to-71 12)))  ; Frequency/Scale (1-12)
        (mode (floor index-0-to-71 12))) ; Operation Type (0-5)
    (case mode
      (0 (list 'sin (list 'multiply k 'x)))           ; Oscillator
      (1 (list 'cos (list 'multiply k 'x)))           ; Phase Shift
      (2 (list 'exp (list 'divide (list 'negate 'x) k))) ; Decay/Entropy
      (3 (list 'divide 1.0d0 (list 'add 'x k)))       ; Singularity/Field
      (4 (list 'log (list 'add 'x k)))                ; Information
      (5 (list 'divide (list 'power 'x 2) k))         ; Potential
      (otherwise 'x))))

;;; ============================================================================
;;; GEOMETRIC TOKENIZER
;;; ============================================================================

(defstruct geometric-tokenizer
  "Tokenizer that maps words to geometric formulas."
  (lexicon (make-hash-table :test 'equal) :type hash-table)
  (domain (make-domain-vector) :type vector)
  (load-success nil :type boolean))

(defun make-domain-vector ()
  "Create the domain vector for embedding (64 points from 0.1 to 2π)."
  (let ((vec (make-array 64 :element-type 'double-float))
        (start 0.1d0)
        (end (* 2.0d0 +pi+)))
    (dotimes (i 64 vec)
      (setf (aref vec i)
            (+ start (* (/ i 63.0d0) (- end start)))))))

(defun load-lexicon (tokenizer filepath)
  "Load lexicon from JSON file.
   Returns T on success, NIL on failure."
  (let ((data (load-json-file filepath)))
    (if data
        (progn
          (parse-lexicon-data tokenizer data)
          (setf (geometric-tokenizer-load-success tokenizer) t)
          (log-message :info "📚 Loaded ~D formulas from ~A"
                      (hash-table-count (geometric-tokenizer-lexicon tokenizer))
                      filepath)
          t)
        (progn
          (log-message :warn "⚠️  Could not load lexicon from ~A" filepath)
          nil))))

(defun parse-lexicon-data (tokenizer data)
  "Parse lexicon data and populate the tokenizer's lexicon.
   Handles both nested (level-based) and flat structures."
  (let ((lexicon (geometric-tokenizer-lexicon tokenizer)))
    (cond
      ;; Check if nested structure (levels -> words)
      ((and (consp data)
            (consp (cdr (first data)))
            (or (assoc :words (cdr (first data)))
                (assoc :constructs (cdr (first data)))))
       ;; Nested structure
       (dolist (level-entry data)
         (let ((level-data (cdr level-entry)))
           (parse-level-words lexicon (cdr (assoc :words level-data)))
           (parse-level-words lexicon (cdr (assoc :constructs level-data))))))
      
      ;; Flat structure
      (t
       (parse-level-words lexicon data)))))

(defun parse-level-words (lexicon words-alist)
  "Parse words from an alist and add to lexicon."
  (dolist (word-entry words-alist)
    (let* ((word (car word-entry))
           (word-data (cdr word-entry))
           (formula (or (cdr (assoc :formula word-data))
                       (cdr (assoc :f word-data)))))
      (when formula
        (setf (gethash (string word) lexicon) formula)))))

(defun get-gematria (text)
  "Compute gematria (sum of character codes) of text."
  (string-gematria text))

(defun word-to-formula (tokenizer word)
  "Get formula for a word.
   If word is in lexicon, return that formula.
   Otherwise, use SH-72 basis fallback."
  (let ((lexicon (geometric-tokenizer-lexicon tokenizer)))
    (or (gethash word lexicon)
        ;; Fallback: use SH-72 basis
        (let* ((gem (get-gematria word))
               (sh-idx (mod gem 72)))
          (get-sh-basis-formula sh-idx)))))

(defun embed-word (tokenizer word)
  "Embed a word as a 64D vector by evaluating its formula.
   Returns a normalized vector."
  (let* ((formula (word-to-formula tokenizer word))
         (domain (geometric-tokenizer-domain tokenizer))
         (vec (make-array 64 :element-type 'double-float)))
    
    ;; Evaluate formula at each domain point
    (dotimes (i 64)
      (let ((x-val (aref domain i)))
        (handler-case
            (setf (aref vec i) (coerce (evaluate-formula formula x-val) 'double-float))
          (error (e)
            (log-message :debug "Error evaluating formula for ~A at x=~A: ~A" word x-val e)
            (setf (aref vec i) 0.0d0)))))
    
    ;; Normalize
    (normalize-vector vec)))

(defun evaluate-formula (formula x-val)
  "Evaluate a formula (S-expression) with x bound to x-val.
   This is a simplified evaluator for geometric formulas."
  (cond
    ;; Number
    ((numberp formula) formula)
    
    ;; Variable 'x
    ((eq formula 'x) x-val)
    
    ;; List (function application)
    ((consp formula)
     (let ((op (first formula))
           (args (rest formula)))
       (case op
         (sin (sin (evaluate-formula (first args) x-val)))
         (cos (cos (evaluate-formula (first args) x-val)))
         (exp (exp (min (evaluate-formula (first args) x-val) 100.0d0)))
         (log (safe-log (evaluate-formula (first args) x-val)))
         (sqrt (safe-sqrt (evaluate-formula (first args) x-val)))
         (abs (abs (evaluate-formula (first args) x-val)))
         (negate (- (evaluate-formula (first args) x-val)))
         (add (+ (evaluate-formula (first args) x-val)
                 (evaluate-formula (second args) x-val)))
         (subtract (- (evaluate-formula (first args) x-val)
                      (evaluate-formula (second args) x-val)))
         (multiply (* (evaluate-formula (first args) x-val)
                      (evaluate-formula (second args) x-val)))
         (divide (safe-divide (evaluate-formula (first args) x-val)
                             (evaluate-formula (second args) x-val)))
         (power (expt (abs (evaluate-formula (first args) x-val))
                     (min (abs (evaluate-formula (second args) x-val)) 10.0d0)))
         (otherwise 0.0d0))))
    
    ;; Unknown
    (t 0.0d0)))

(defun resonance (tokenizer word1 word2)
  "Calculate quantum resonance (cosine similarity) between two words."
  (let ((vec1 (embed-word tokenizer word1))
        (vec2 (embed-word tokenizer word2)))
    (dot-product vec1 vec2)))

;;; ============================================================================
;;; TOKENIZER UTILITIES
;;; ============================================================================

(defun tokenizer-vocabulary-size (tokenizer)
  "Get the number of words in the lexicon."
  (hash-table-count (geometric-tokenizer-lexicon tokenizer)))

(defun tokenizer-has-word-p (tokenizer word)
  "Check if a word is in the lexicon."
  (gethash word (geometric-tokenizer-lexicon tokenizer)))

(defun tokenizer-words (tokenizer)
  "Get all words in the lexicon."
  (kb-keys (geometric-tokenizer-lexicon tokenizer)))
