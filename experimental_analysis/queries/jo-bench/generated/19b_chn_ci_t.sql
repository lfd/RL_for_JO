SELECT * FROM title AS t, cast_info AS ci, char_name AS chn WHERE ci.note = '(voice)' AND t.production_year BETWEEN 2007 AND 2008 AND t.title LIKE '%Kung%Fu%Panda%' AND t.id = ci.movie_id AND ci.movie_id = t.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;