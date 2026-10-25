;;;; tests/test-geometric-tokenizer.lisp
;;;; Tests for geometric tokenizer

(in-package #:lisp-cognito-philosopher-tests)

(test gematria-computation
  "Test gematria computation"
  (is (numberp (get-gematria "test")))
  (is (> (get-gematria "test") 0))
  (is (= (get-gematria "abc") (get-gematria "abc"))))

(test sh-basis-formula
  "Test SH-72 basis formula generation"
  (dotimes (i 72)
    (let ((formula (get-sh-basis-formula i)))
      (is (not (null formula))))))

(test word-embedding
  "Test word embedding"
  (let ((tokenizer (make-geometric-tokenizer)))
    (let ((vec (embed-word tokenizer "test")))
      (is (vectorp vec))
      (is (= (length vec) 64))
      ;; Check normalization
      (let ((norm (vector-norm vec)))
        (is (or (< (abs (- norm 1.0)) 0.01) ; Normalized
                (< norm 0.01))))))) ; Or zero vector

(test resonance-computation
  "Test resonance between words"
  (let ((tokenizer (make-geometric-tokenizer)))
    (let ((res (resonance tokenizer "test" "test")))
      (is (numberp res))
      ;; Self-resonance should be high (close to 1 for normalized vectors)
      (is (or (> res 0.9) (< (abs res) 0.1)))))) ; High or zero
