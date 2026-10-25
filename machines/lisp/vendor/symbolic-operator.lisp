;;;; symbolic-operator.lisp
;;;; Symbolic computation engine for AST generation and evaluation

(in-package #:lisp-cognito-philosopher)

;;; ============================================================================
;;; SYMBOLIC OPERATORS
;;; ============================================================================

(defparameter *unary-operators*
  '(identity negate square sqrt sin cos exp log abs inv)
  "Available unary operators")

(defparameter *binary-operators*
  '(add subtract multiply divide power)
  "Available binary operators")

(defparameter *constants*
  (list 0.0d0 1.0d0 2.0d0 +pi+ +e+ +phi+)
  "Available constants")

;;; ============================================================================
;;; OPERATOR APPLICATION
;;; ============================================================================

(defun apply-operator (op &rest args)
  "Apply a symbolic operator to arguments.
   Returns the result or signals an error."
  (case op
    ;; Unary operators
    (identity (first args))
    (negate (- (first args)))
    (square (* (first args) (first args)))
    (sqrt (safe-sqrt (first args)))
    (sin (sin (first args)))
    (cos (cos (first args)))
    (exp (exp (min (first args) 100.0d0))) ; Prevent overflow
    (log (safe-log (first args)))
    (abs (abs (first args)))
    (inv (safe-divide 1.0d0 (first args)))
    
    ;; Binary operators
    (add (+ (first args) (second args)))
    (subtract (- (first args) (second args)))
    (multiply (* (first args) (second args)))
    (divide (safe-divide (first args) (second args)))
    (power (expt (abs (first args)) (min (abs (second args)) 10.0d0))) ; Prevent overflow
    
    (otherwise
     (error "Unknown operator: ~A" op))))

;;; ============================================================================
;;; AST GENERATION
;;; ============================================================================

(defun generate-random-ast (&optional (depth 0) (max-depth 3))
  "Generate a random symbolic AST.
   Returns an S-expression representing the formula.
   
   AST format:
   - (:const value)
   - (:var symbol)
   - (operator arg1 arg2 ...)"
  (if (or (>= depth max-depth) (< (deterministic-random 1.0) 0.3))
      ;; Leaf node: constant or variable
      (if (< (deterministic-random 1.0) 0.5)
          (if (< (deterministic-random 1.0) 0.5)
              (list :var 'x)
              (list :const (deterministic-random 10.0)))
          (list :const (random-element *constants*)))
      ;; Internal node: operator
      (if (< (deterministic-random 1.0) 0.6)
          ;; Unary operator
          (list (random-element *unary-operators*)
                (generate-random-ast (1+ depth) max-depth))
          ;; Binary operator
          (list (random-element *binary-operators*)
                (generate-random-ast (1+ depth) max-depth)
                (generate-random-ast (1+ depth) max-depth)))))

(defun random-element (list)
  "Return a random element from a list."
  (nth (deterministic-random (length list)) list))

;;; ============================================================================
;;; AST EVALUATION
;;; ============================================================================

(defun evaluate-ast (ast bindings)
  "Evaluate an AST with variable bindings.
   
   bindings: alist of (symbol . value) pairs
   Returns: numeric result"
  (cond
    ;; Constant
    ((and (consp ast) (eq (first ast) :const))
     (second ast))
    
    ;; Variable
    ((and (consp ast) (eq (first ast) :var))
     (let ((var-name (second ast)))
       (or (cdr (assoc var-name bindings))
           (error "Unbound variable: ~A" var-name))))
    
    ;; Operator application
    ((consp ast)
     (let ((op (first ast))
           (args (rest ast)))
       (apply #'apply-operator op (mapcar (lambda (arg) (evaluate-ast arg bindings)) args))))
    
    ;; Atom (treat as constant)
    (t ast)))

;;; ============================================================================
;;; AST SIMPLIFICATION
;;; ============================================================================

(defun simplify-ast (ast)
  "Simplify an AST using algebraic rules.
   This is a basic implementation - could be extended."
  (cond
    ;; Already simplified (leaf nodes)
    ((atom ast) ast)
    ((eq (first ast) :const) ast)
    ((eq (first ast) :var) ast)
    
    ;; Simplify recursively
    (t
     (let ((op (first ast))
           (simplified-args (mapcar #'simplify-ast (rest ast))))
       (apply-simplification-rules op simplified-args)))))

(defun apply-simplification-rules (op args)
  "Apply simplification rules to an operator and its arguments."
  (case op
    ;; Identity rules
    (identity (first args))
    
    ;; Addition rules
    (add
     (cond
       ;; 0 + x = x
       ((and (consp (first args)) (eq (first (first args)) :const)
             (zerop (second (first args))))
        (second args))
       ;; x + 0 = x
       ((and (consp (second args)) (eq (first (second args)) :const)
             (zerop (second (second args))))
        (first args))
       ;; Both constants
       ((and (consp (first args)) (eq (first (first args)) :const)
             (consp (second args)) (eq (first (second args)) :const))
        (list :const (+ (second (first args)) (second (second args)))))
       (t (cons op args))))
    
    ;; Multiplication rules
    (multiply
     (cond
       ;; 0 * x = 0
       ((and (consp (first args)) (eq (first (first args)) :const)
             (zerop (second (first args))))
        (list :const 0.0d0))
       ;; x * 0 = 0
       ((and (consp (second args)) (eq (first (second args)) :const)
             (zerop (second (second args))))
        (list :const 0.0d0))
       ;; 1 * x = x
       ((and (consp (first args)) (eq (first (first args)) :const)
             (= 1.0d0 (second (first args))))
        (second args))
       ;; x * 1 = x
       ((and (consp (second args)) (eq (first (second args)) :const)
             (= 1.0d0 (second (second args))))
        (first args))
       ;; Both constants
       ((and (consp (first args)) (eq (first (first args)) :const)
             (consp (second args)) (eq (first (second args)) :const))
        (list :const (* (second (first args)) (second (second args)))))
       (t (cons op args))))
    
    ;; Default: no simplification
    (otherwise (cons op args))))

;;; ============================================================================
;;; AST TO STRING CONVERSION
;;; ============================================================================

(defun ast-to-string (ast)
  "Convert an AST to a human-readable string."
  (cond
    ((atom ast) (format nil "~A" ast))
    ((eq (first ast) :const) (format nil "~,2F" (second ast)))
    ((eq (first ast) :var) (format nil "~A" (second ast)))
    (t
     (let ((op (first ast))
           (args (rest ast)))
       (case op
         ;; Unary operators
         ((identity negate square sqrt sin cos exp log abs inv)
          (format nil "~A(~A)" op (ast-to-string (first args))))
         ;; Binary operators - infix notation
         ((add subtract multiply divide power)
          (format nil "(~A ~A ~A)"
                  (ast-to-string (first args))
                  (operator-symbol op)
                  (ast-to-string (second args))))
         (otherwise
          (format nil "~A(~{~A~^, ~})" op (mapcar #'ast-to-string args))))))))

(defun operator-symbol (op)
  "Get the infix symbol for a binary operator."
  (case op
    (add "+")
    (subtract "-")
    (multiply "*")
    (divide "/")
    (power "^")
    (otherwise (string op))))

;;; ============================================================================
;;; AST UTILITIES
;;; ============================================================================

(defun ast-variables (ast)
  "Extract all variables from an AST."
  (cond
    ((atom ast) nil)
    ((eq (first ast) :var) (list (second ast)))
    ((eq (first ast) :const) nil)
    (t (remove-duplicates (mapcan #'ast-variables (rest ast))))))

(defun ast-depth (ast)
  "Compute the depth of an AST."
  (cond
    ((atom ast) 0)
    ((member (first ast) '(:const :var)) 0)
    (t (1+ (reduce #'max (mapcar #'ast-depth (rest ast)) :initial-value 0)))))

(defun ast-size (ast)
  "Compute the size (number of nodes) of an AST."
  (cond
    ((atom ast) 1)
    (t (1+ (reduce #'+ (mapcar #'ast-size (rest ast)) :initial-value 0)))))
