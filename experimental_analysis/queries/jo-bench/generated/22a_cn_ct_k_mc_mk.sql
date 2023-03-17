SELECT * FROM movie_keyword AS mk, movie_companies AS mc, company_type AS ct, company_name AS cn, keyword AS k WHERE cn.country_code != '[us]' AND k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cn.id = mc.company_id AND mc.company_id = cn.id;