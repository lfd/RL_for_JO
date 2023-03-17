SELECT * FROM title AS t, cast_info AS ci, aka_name AS an, movie_info AS mi, char_name AS chn WHERE ci.note = '(voice)' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%2007%' OR mi.info LIKE 'USA:%2008%') AND t.production_year BETWEEN 2007 AND 2008 AND t.title LIKE '%Kung%Fu%Panda%' AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;