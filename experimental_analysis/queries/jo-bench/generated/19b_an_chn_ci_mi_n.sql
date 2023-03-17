SELECT * FROM cast_info AS ci, char_name AS chn, movie_info AS mi, name AS n, aka_name AS an WHERE ci.note = '(voice)' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%2007%' OR mi.info LIKE 'USA:%2008%') AND n.gender = 'f' AND n.name LIKE '%Angel%' AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND n.id = an.person_id AND an.person_id = n.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;