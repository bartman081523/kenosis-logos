;;;; dialectic-macros.lisp
;;;; Self-modifying dialectic system using Lisp macros

(in-package #:lisp-cognito-philosopher)

;;; ============================================================================
;;; DIALECTIC COMPONENT STORAGE
;;; ============================================================================

(defparameter *thesis-generators* (make-hash-table :test 'equal)
  "Storage for thesis generators")

(defparameter *antithesis-generators* (make-hash-table :test 'equal)
  "Storage for antithesis generators")

(defparameter *synthesis-generators* (make-hash-table :test 'equal)
  "Storage for synthesis generators")

;;; ============================================================================
;;; THESIS MACRO
;;; ============================================================================

(defmacro defthesis (name params &body body)
  "Define a thesis generator.
   
   Example:
   (defthesis energy-thesis ()
     (generate-random-ast 2))"
  `(progn
     (defun ,name ,params ,@body)
     (setf (gethash ,(string name) *thesis-generators*)
           (list :name ',name
                 :params ',params
                 :body ',body
                 :generation 0))
     ',name))

;;; ============================================================================
;;; ANTITHESIS MACRO
;;; ============================================================================

(defmacro defantithesis (name params &body body)
  "Define an antithesis generator.
   
   Example:
   (defantithesis energy-antithesis (thesis-result)
     (list 'negate thesis-result))"
  `(progn
     (defun ,name ,params ,@body)
     (setf (gethash ,(string name) *antithesis-generators*)
           (list :name ',name
                 :params ',params
                 :body ',body
                 :generation 0))
     ',name))

;;; ============================================================================
;;; SYNTHESIS MACRO
;;; ============================================================================

(defmacro defsynthesis (name thesis-name antithesis-name)
  "Define a synthesis that combines thesis and antithesis.
   
   Example:
   (defsynthesis energy-synthesis energy-thesis energy-antithesis)"
  `(progn
     (defun ,name ()
       (let* ((thesis-result (,thesis-name))
              (antithesis-result (,antithesis-name thesis-result)))
         (synthesize-results thesis-result antithesis-result)))
     (setf (gethash ,(string name) *synthesis-generators*)
           (list :name ',name
                 :thesis ',thesis-name
                 :antithesis ',antithesis-name
                 :generation 0))
     ',name))

(defun synthesize-results (thesis antithesis)
  "Synthesize thesis and antithesis into a unified result.
   This is a simple implementation - could be more sophisticated."
  (cond
    ;; Both are ASTs
    ((and (consp thesis) (consp antithesis))
     (list 'add thesis antithesis))
    ;; Both are numbers
    ((and (numberp thesis) (numberp antithesis))
     (/ (+ thesis antithesis) 2.0))
    ;; Mixed - convert to list
    (t (list thesis antithesis))))

;;; ============================================================================
;;; SELF-MODIFICATION
;;; ============================================================================

(defmacro modify-generator (target-name new-body)
  "Modify an existing generator's body.
   This is the self-modification primitive.
   
   Example:
   (modify-generator energy-thesis
     (generate-random-ast 3))"
  `(progn
     (let ((generator (or (gethash ,target-name *thesis-generators*)
                         (gethash ,target-name *antithesis-generators*))))
       (when generator
         (let ((name (getf generator :name))
               (params (getf generator :params)))
           ;; Redefine the function with new body
           (eval `(defun ,name ,params ,',new-body))
           ;; Update generation counter
           (incf (getf generator :generation))
           (log-message :info "🔄 Modified ~A (generation ~D)" 
                       ,target-name (getf generator :generation))
           t)))))

;;; ============================================================================
;;; DIALECTIC UTILITIES
;;; ============================================================================

(defun list-thesis-generators ()
  "List all defined thesis generators."
  (loop for name being the hash-keys of *thesis-generators*
        collect name))

(defun list-antithesis-generators ()
  "List all defined antithesis generators."
  (loop for name being the hash-keys of *antithesis-generators*
        collect name))

(defun list-synthesis-generators ()
  "List all defined synthesis generators."
  (loop for name being the hash-keys of *synthesis-generators*
        collect name))

(defun get-generator-info (name)
  "Get information about a generator."
  (or (gethash name *thesis-generators*)
      (gethash name *antithesis-generators*)
      (gethash name *synthesis-generators*)))

(defun generator-generation (name)
  "Get the generation number of a generator."
  (let ((info (get-generator-info name)))
    (if info
        (getf info :generation)
        0)))
