SELECT * FROM cast_info AS ci, role_type AS rt, title AS t, movie_companies AS mc, aka_name AS an WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND rt.role = 'actress' AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND ci.role_id = rt.id AND rt.id = ci.role_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id;