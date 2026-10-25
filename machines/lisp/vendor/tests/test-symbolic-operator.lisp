;;;; tests/test-symbolic-operator.lisp
;;;; Tests for symbolic operator

(in-package #:lisp-cognito-philosopher-tests)

(test ast-generation
  "Test AST generation"
  (let ((ast (generate-random-ast)))
    (is (not (null ast)))
    (is (or (consp ast) (numberp ast)))))

(test ast-evaluation
  "Test AST evaluation"
  (let ((ast '(add (:const 2.0) (:var x))))
    (let ((result (evaluate-ast ast '((x . 3.0)))))
      (is (numberp result))
      (is (= result 5.0)))))

(test ast-simplification
  "Test AST simplification"
  ;; 0 + x = x
  (let ((ast '(add (:const 0.0) (:var x))))
    (let ((simplified (simplify-ast ast)))
      (is (equal simplified '(:var x)))))
  
  ;; x * 1 = x
  (let ((ast '(multiply (:var x) (:const 1.0))))
    (let ((simplified (simplify-ast ast)))
      (is (equal simplified '(:var x))))))

(test ast-to-string
  "Test AST to string conversion"
  (let ((ast '(add (:const 2.0) (:var x))))
    (let ((str (ast-to-string ast)))
      (is (stringp str))
      (is (> (length str) 0)))))

(test operator-application
  "Test operator application"
  (is (= (apply-operator 'add 2.0 3.0) 5.0))
  (is (= (apply-operator 'multiply 2.0 3.0) 6.0))
  (is (= (apply-operator 'negate 5.0) -5.0))
  (is (= (apply-operator 'square 3.0) 9.0)))
