SELECT * FROM cast_info AS ci, complete_cast AS cc, comp_cast_type AS cct1 WHERE cct1.kind = 'cast' AND ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;