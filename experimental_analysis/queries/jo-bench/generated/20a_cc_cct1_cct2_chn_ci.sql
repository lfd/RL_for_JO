SELECT * FROM cast_info AS ci, complete_cast AS cc, comp_cast_type AS cct2, comp_cast_type AS cct1, char_name AS chn WHERE cct1.kind = 'cast' AND cct2.kind LIKE '%complete%' AND chn.name NOT LIKE '%Sherlock%' AND (chn.name LIKE '%Tony%Stark%' OR chn.name LIKE '%Iron%Man%') AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;