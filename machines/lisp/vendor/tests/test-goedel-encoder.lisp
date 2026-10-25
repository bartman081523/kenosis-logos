;;;; tests/test-goedel-encoder.lisp
;;;; Tests for Gödel encoder

(in-package #:lisp-cognito-philosopher-tests)

(test goedel-encoding
  "Test Gödel encoding produces consistent numbers"
  (let ((expr '(add (multiply 2 x) 1)))
    (is (numberp (goedel-encode expr)))
    (is (= (goedel-encode expr) (goedel-encode expr)))
    (is (not (= (goedel-encode expr) (goedel-encode '(subtract x 1)))))))

(test structural-signature
  "Test structural signature extraction"
  (let ((expr '(add (multiply 2 x) 1)))
    (let ((sig (compute-structural-signature expr)))
      (is (hash-table-p sig))
      (is (> (gethash "depth" sig) 0))
      (is (> (gethash "size" sig) 0)))))

(test gap-computation
  "Test Gödel Gap computation"
  (let ((pred #(1.0 2.0 3.0))
        (actual #(1.1 2.1 3.1)))
    (let ((gap (compute-goedel-gap pred actual)))
      (is (numberp gap))
      (is (>= gap 0.0))
      (is (<= gap 1.0)))))

(test gap-interpretation
  "Test gap interpretation"
  (is (eq (interpret-gap 0.05) :excellent-self-knowledge))
  (is (eq (interpret-gap 0.2) :good-self-knowledge))
  (is (eq (interpret-gap 0.4) :moderate-blind-spot))
  (is (eq (interpret-gap 0.6) :significant-blind-spot))
  (is (eq (interpret-gap 0.9) :severe-blind-spot)))
