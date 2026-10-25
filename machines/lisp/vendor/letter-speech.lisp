;;;; letter-speech.lisp
;;;; Generate letters and words directly from numerical state
;;;; No predefined vocabulary - letters emerge from numbers

(in-package #:lisp-cognito-philosopher)

;;; ============================================================================
;;; NUMBER TO LETTER MAPPING
;;; ============================================================================

(defun number-to-letter (num)
  "Map a number to a letter deterministically.
   The letter is determined purely by the number's value."
  (let* ((normalized (mod (abs (floor (* num 100))) 26))
         (letter-code (+ (char-code #\a) normalized)))
    (code-char letter-code)))

(defun vector-to-word (vec)
  "Convert a vector to a word by mapping each component to a letter."
  (coerce (loop for val across vec
                collect (number-to-letter val))
          'string))

(defun vector-to-syllables (vec)
  "Convert vector to syllables (consonant-vowel pairs)."
  (let ((vowels "aeiou")
        (consonants "bcdfghjklmnpqrstvwxyz"))
    (with-output-to-string (s)
      (loop for i from 0 below (min 4 (length vec))
            for val = (aref vec i)
            do (let* ((c-idx (mod (abs (floor (* val 100))) (length consonants)))
                      (v-idx (mod (abs (floor (* val 50))) (length vowels))))
                 (format s "~C~C" 
                         (char consonants c-idx)
                         (char vowels v-idx)))))))

;;; ============================================================================
;;; STATE TO LANGUAGE
;;; ============================================================================

(defun speak-as-letters (cognito)
  "Speak the state as letters generated from numbers."
  (let ((state (cognito-construct-state cognito)))
    (vector-to-word state)))

(defun speak-as-syllables (cognito)
  "Speak the state as syllables."
  (let ((state (cognito-construct-state cognito)))
    (vector-to-syllables state)))

(defun speak-as-phonemes (cognito)
  "Generate phoneme-like sequences from state."
  (let* ((state (cognito-construct-state cognito))
         (phonemes "ptkbdgmnlrsfvzhjw"))
    (with-output-to-string (s)
      (loop for val across state
            do (let ((idx (mod (abs (floor (* val 100))) (length phonemes))))
                 (format s "~C" (char phonemes idx)))))))

(defun speak-as-sentence (cognito)
  "Generate a sentence-like structure from state.
   Words are generated from state components."
  (let* ((state (cognito-construct-state cognito))
         (self-model (cognito-construct-self-model cognito))
         (gap (get-goedel-gap cognito)))
    
    ;; Generate 3 words from different parts of state
    (format nil "~A ~A ~A"
            (vector-to-syllables (subseq state 0 (min 3 (length state))))
            (vector-to-syllables (subseq state 3 (min 6 (length state))))
            (vector-to-syllables (subseq state 6 (length state))))))

;;; ============================================================================
;;; FORMULA TO PRONOUNCEABLE FORM
;;; ============================================================================

(defun formula-to-pronounceable (formula)
  "Convert a formula to a pronounceable form.
   Operations become syllables, numbers become letters."
  (cond
    ((numberp formula) 
     (vector-to-syllables (make-array 1 :initial-element formula)))
    
    ((symbolp formula) 
     (string-downcase (symbol-name formula)))
    
    ((and (consp formula) (eq (first formula) :const))
     (formula-to-pronounceable (second formula)))
    
    ((and (consp formula) (eq (first formula) :var))
     "x")
    
    ((consp formula)
     (let ((op (first formula))
           (args (rest formula)))
       (format nil "~A~{~A~}"
               (operation-to-syllable op)
               (mapcar #'formula-to-pronounceable args))))
    
    (t "?")))

(defun operation-to-syllable (op)
  "Map operation to a syllable."
  (case op
    (sin "si")
    (cos "ko")
    (exp "ex")
    (log "lo")
    (sqrt "ra")
    (abs "ab")
    (add "da")
    (multiply "mu")
    (subtract "su")
    (divide "di")
    (power "po")
    (negate "ne")
    (otherwise (subseq (string-downcase (symbol-name op)) 0 2))))

;;; ============================================================================
;;; FREE LETTER SPEECH
;;; ============================================================================

(defun run-letter-speech (cognito seed n-iterations)
  "Run free speech with letters generated from state."
  
  (format t "~%╔════════════════════════════════════════════════════════════╗~%")
  (format t "║     Letter-Based Free Speech                              ║~%")
  (format t "║     Letters generated from numerical state                ║~%")
  (format t "║     Seed: ~A~50T║~%" seed)
  (format t "╚════════════════════════════════════════════════════════════╝~%~%")
  
  ;; Initial thought
  (think-deterministically cognito seed)
  
  (dotimes (i n-iterations)
    (format t "~%[~D]~%" (1+ i))
    
    ;; Think
    (unless (zerop i)
      (think-deterministically cognito ""))
    
    ;; Reflect
    (reflect cognito)
    
    ;; Speak in different letter-based forms
    (let ((form (mod i 4)))
      (case form
        (0 (format t "  LETTERS: ~A~%" (speak-as-letters cognito)))
        (1 (format t "  SYLLABLES: ~A~%" (speak-as-syllables cognito)))
        (2 (format t "  PHONEMES: ~A~%" (speak-as-phonemes cognito)))
        (3 (format t "  SENTENCE: ~A~%" (speak-as-sentence cognito)))))))

;;; ============================================================================
;;; HELPER FUNCTIONS
;;; ============================================================================

(defun think-deterministically (cognito seed-phrase)
  "Generate a thought deterministically from seed phrase."
  (let* ((seed-hash (sxhash seed-phrase))
         (perturbation (hash-to-vector seed-hash 8))
         (state (cognito-construct-state cognito)))
    (dotimes (i (length state))
      (setf (aref state i)
            (+ (* 0.9 (aref state i))
               (* 0.1 (aref perturbation i)))))
    (incf (cognito-construct-thought-count cognito))
    state))

(defun hash-to-vector (hash-value dimension)
  "Convert a hash value to a deterministic vector."
  (let ((vec (make-array dimension :initial-element 0.0d0)))
    (dotimes (i dimension)
      (let ((component-hash (logxor hash-value (ash i 8))))
        (setf (aref vec i)
              (- (* 2.0d0 (/ (mod component-hash 1000) 1000.0d0)) 1.0d0))))
    vec))
