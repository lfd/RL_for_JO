SELECT * FROM company_type AS ct, movie_companies AS mc, complete_cast AS cc, comp_cast_type AS cct2, title AS t WHERE cct2.kind = 'complete' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND t.production_year > 2005 AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;