SELECT * FROM title AS t, cast_info AS ci, name AS n, role_type AS rt WHERE ci.note = '(voice)' AND n.gender = 'f' AND n.name LIKE '%Angel%' AND rt.role = 'actress' AND t.production_year BETWEEN 2007 AND 2010 AND ci.movie_id = t.id AND t.id = ci.movie_id AND ci.role_id = rt.id AND rt.id = ci.role_id AND n.id = ci.person_id AND ci.person_id = n.id;