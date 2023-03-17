SELECT * FROM comp_cast_type AS cct2, movie_companies AS mc, complete_cast AS cc, comp_cast_type AS cct1, title AS t, kind_type AS kt, company_type AS ct WHERE cct1.kind = 'cast' AND cct2.kind = 'complete' AND kt.kind IN ('movie', 'episode') AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND t.production_year > 2005 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;