SELECT * FROM name AS n, cast_info AS ci, char_name AS chn, role_type AS rt, title AS t, aka_name AS an WHERE ci.note = '(voice)' AND n.gender = 'f' AND n.name LIKE '%Angel%' AND rt.role = 'actress' AND t.production_year BETWEEN 2007 AND 2008 AND t.title LIKE '%Kung%Fu%Panda%' AND t.id = ci.movie_id AND ci.movie_id = t.id AND n.id = ci.person_id AND ci.person_id = n.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND n.id = an.person_id AND an.person_id = n.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;