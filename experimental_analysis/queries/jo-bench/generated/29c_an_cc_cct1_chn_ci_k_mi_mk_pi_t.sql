SELECT * FROM comp_cast_type AS cct1, person_info AS pi, keyword AS k, char_name AS chn, cast_info AS ci, aka_name AS an, movie_keyword AS mk, movie_info AS mi, title AS t, complete_cast AS cc WHERE cct1.kind = 'cast' AND ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND k.keyword = 'computer-animation' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND t.production_year BETWEEN 2000 AND 2010 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND ci.person_id = pi.person_id AND pi.person_id = ci.person_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;