SELECT * FROM movie_companies AS mc, cast_info AS ci, role_type AS rt, title AS t, aka_name AS an1 WHERE ci.note = '(voice: English version)' AND mc.note LIKE '%(Japan)%' AND mc.note NOT LIKE '%(USA)%' AND rt.role = 'actress' AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ci.role_id = rt.id AND rt.id = ci.role_id AND an1.person_id = ci.person_id AND ci.person_id = an1.person_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;