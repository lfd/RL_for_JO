SELECT * FROM title AS t, role_type AS rt, cast_info AS ci WHERE ci.note = '(voice)' AND rt.role = 'actress' AND t.production_year BETWEEN 2007 AND 2008 AND t.title LIKE '%Kung%Fu%Panda%' AND t.id = ci.movie_id AND ci.movie_id = t.id AND rt.id = ci.role_id AND ci.role_id = rt.id;