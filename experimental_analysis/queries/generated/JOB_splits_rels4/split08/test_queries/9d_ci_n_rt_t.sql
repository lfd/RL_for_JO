SELECT * FROM name AS n, role_type AS rt, cast_info AS ci, title AS t WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND n.gender = 'f' AND rt.role = 'actress' AND ci.movie_id = t.id AND t.id = ci.movie_id AND ci.role_id = rt.id AND rt.id = ci.role_id AND n.id = ci.person_id AND ci.person_id = n.id;