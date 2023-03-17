SELECT * FROM title AS t, cast_info AS ci, role_type AS rt, name AS n WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND n.gender = 'f' AND rt.role = 'actress' AND t.production_year > 2000 AND t.id = ci.movie_id AND ci.movie_id = t.id AND n.id = ci.person_id AND ci.person_id = n.id AND rt.id = ci.role_id AND ci.role_id = rt.id;