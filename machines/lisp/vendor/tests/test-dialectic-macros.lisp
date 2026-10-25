;;;; tests/test-dialectic-macros.lisp
;;;; Tests for dialectic macros

(in-package #:lisp-cognito-philosopher-tests)

(test thesis-definition
  "Test thesis definition"
  (defthesis test-thesis-1 ()
    (list :const 42.0))
  (is (fboundp 'test-thesis-1))
  (is (equal (test-thesis-1) '(:const 42.0))))

(test antithesis-definition
  "Test antithesis definition"
  (defantithesis test-antithesis-1 (thesis-result)
    (list 'negate thesis-result))
  (is (fboundp 'test-antithesis-1))
  (let ((result (test-antithesis-1 '(:const 5.0))))
    (is (consp result))
    (is (eq (first result) 'negate))))

(test synthesis-definition
  "Test synthesis definition"
  (defthesis test-thesis-2 ()
    (list :const 10.0))
  (defantithesis test-antithesis-2 (t-result)
    (list 'negate t-result))
  (defsynthesis test-synthesis-2 test-thesis-2 test-antithesis-2)
  (is (fboundp 'test-synthesis-2))
  (let ((result (test-synthesis-2)))
    (is (not (null result)))))

(test generator-modification
  "Test generator self-modification"
  (defthesis test-thesis-3 ()
    (list :const 1.0))
  (let ((original-result (test-thesis-3)))
    (modify-generator "TEST-THESIS-3" (list :const 2.0))
    (let ((modified-result (test-thesis-3)))
      (is (not (equal original-result modified-result))))))
